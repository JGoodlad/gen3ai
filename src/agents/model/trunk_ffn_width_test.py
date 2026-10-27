"""B-FFNW (`agents.model.trunk_ffn_width`, `--trunk-ffn-dim N`, `--grow trunk_ffn_dim=N`; T69, the N4 lever,
`designs/endstate/design_n4_capacity_screen.md` §2): the model side.

The parent is the toy network (`adopted_levers_test.TOY_ARGV`, the X34 toy: T*'s architecture, FFN 256) after ONE real
AdamW step, built through SB3 (`build_fresh_model`), CPU, one torch thread, fp32 eager; the growth is N4's, 256 -> 1024.

Each test fails on the revert it names:
* W-1 the grown child is its parent's function bit for bit (the live policy AND the fork-time gate's CPU builds) and a
  fresh 1024 build's state dict (keys, order, shapes, and the branches' init bytes) — fails if the branch's ``linear2``
  is not zero (the gate refuses), if the branch is not called from ``round.ffn`` (the fresh child's forward differs from
  its own state), or if the grower is reverted (no ``ffn_extra`` keys);
* W-2 the optimizer: every parent parameter's AdamW state carried by identity, the six new tensors in the one param
  group with NO state, and after one more step their own step count (fresh bias correction);
* W-3 256 builds nothing, and a 1024 build moves no other parameter's initial bytes (the private seed);
* W-4 a save / load round trip through the anchors' bare loader (`load_checkpoint_strict`) and `load_model_snapshot` at
  the child's recorded version: the same function, the field recorded, the migration of a pre-v164 config to 256;
* W-5 depth growth on a WIDE parent and depth + width at one fork both equal a fresh child;
* W-6 the refusals (parse, narrowing, widening a wide parent, the per-kind-expert pair).
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from agents.model import growth as G
from utils.torch_state_guard import torch_globals

FIELDS = {"trunk_ffn_dim": {"parent": 256, "child": 1024}}
#: Per round: linear1 128 x 768 + 768 (bias) = 99,072; linear2 768 x 128, no bias = 98,304. The toy has 2 rounds.
NEW_PER_ROUND = 128 * 768 + 768 + 768 * 128
NEW_ELEMENTS = 2 * NEW_PER_ROUND                      # 394,752 (design_n4_capacity_screen.md §2's table)


def _toy(trunk_layers: int = 2, trunk_ffn_dim: int = 256) -> Any:
    from agents.model.adopted_levers_test import _toy_args
    from main.fresh_checkpoint import build_fresh_model

    args = copy.deepcopy(_toy_args())
    args.trunk_layers = trunk_layers
    args.trunk_ffn_dim = trunk_ffn_dim
    model, _args, _pk = build_fresh_model(0, args=args)
    return model


@pytest.fixture(scope="module")
def rows() -> Tuple[np.ndarray, np.ndarray]:
    from agents.model.growth_test import _rows
    return _rows()


def _parent(rows: Tuple[np.ndarray, np.ndarray], **kw: Any) -> Any:
    from agents.model.growth_test import _step
    model = _toy(**kw)
    _step(model, *rows)
    return model


def _fwd(model: Any, rows: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
    model.policy.set_training_mode(False)
    return G.forward_record(model.policy, *rows)


def _same(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> bool:
    return bool(np.array_equal(a["logits"], b["logits"]) and np.array_equal(a["values"], b["values"]))


def test_w1_the_grown_toy_is_its_parents_function_bit_for_bit_and_a_fresh_1024_build(rows) -> None:
    with torch_globals(num_threads=1):
        model = _parent(rows)
        before = _fwd(model, rows)
        keys_before = list(model.policy.state_dict())
        report = G.apply_growth(model, FIELDS, rows=rows)
        after = _fwd(model, rows)
        fresh = _toy(trunk_ffn_dim=1024)
        # the fresh child's forward with the GROWN weights loaded is the grown function (the branch is on the path)
        fresh.policy.load_state_dict(model.policy.state_dict(), strict=True)
        via_fresh = _fwd(fresh, rows)
        fresh_init = _toy(trunk_ffn_dim=1024).policy.state_dict()
    fc = report["function_check"]
    assert fc["bit_identical"] and fc["max_abs_d_logits"] == 0.0 and fc["max_abs_d_value"] == 0.0 and fc["rows"] == 64
    assert _same(before, after) and _same(after, via_fresh)
    sd = model.policy.state_dict()
    assert list(sd) == list(fresh_init) and all(tuple(sd[k].shape) == tuple(fresh_init[k].shape) for k in sd)
    new = [k for k in sd if k not in keys_before]
    assert new and all(".ffn_extra." in k for k in new)
    # the grown branches are the bytes a fresh 1024 build initialises them to (same modules, same private seeds)
    assert all(torch.equal(sd[k], fresh_init[k]) for k in new)
    zero = [k for k in new if k.startswith("features_extractor.") and ".linear2." in k]
    assert len(zero) == 2 and all(float(sd[k].abs().max()) == 0.0 for k in zero)
    assert not any(k.endswith("ffn_extra.linear2.bias") for k in sd)          # one output bias: the base's
    assert report["new_parameters"] == {"tensors": 6, "elements": NEW_ELEMENTS}
    fe = model.policy.features_extractor
    assert fe.trunk_ffn_dim == 1024 and fe.team_transformer.trunk_ffn_dim == 1024 and G.trunk_ffn_dim_of(model.policy) == 1024
    assert model.policy_kwargs["features_extractor_kwargs"]["trunk_ffn_dim"] == 1024


def test_w1b_a_non_identity_branch_is_refused_by_the_fork_time_gate(rows, monkeypatch) -> None:
    """The gate is what makes the zero init load-bearing: a branch whose output is not zero is refused, typed."""
    from agents.model import trunk_ffn_width as W

    real = W.build_ffn_extra

    def nonzero(i: int, n: int) -> Any:
        br = real(i, n)
        if br is not None:
            with torch.no_grad():
                br.linear2.weight.fill_(1e-3)
        return br

    with torch_globals(num_threads=1):
        model = _parent(rows)
        monkeypatch.setattr(W, "build_ffn_extra", nonzero)
        with pytest.raises(G.GrowthError, match="NOT its parent's function"):
            G.apply_growth(model, FIELDS, rows=rows)


def test_w2_the_optimizer_carries_the_parents_adamw_state_and_the_branches_start_fresh(rows) -> None:
    from agents.model.growth_test import _step

    with torch_globals(num_threads=1):
        model = _parent(rows)
        old_opt = model.policy.optimizer
        old_params = list(old_opt.param_groups[0]["params"])
        old_state = {id(p): old_opt.state[p] for p in old_params}
        old_group = {k: v for k, v in old_opt.param_groups[0].items() if k != "params"}
        report = G.apply_growth(model, FIELDS, rows=rows)
        opt = model.policy.optimizer
        new = [p for n, p in model.policy.named_parameters() if ".ffn_extra." in n]
        assert report["optimizer_state"] == {"carried": len(old_params), "no_state": 0, "fresh": 6}
        assert {k: v for k, v in opt.param_groups[0].items() if k != "params"} == old_group
        assert all(opt.state[p] is old_state[id(p)] for p in old_params)
        group = opt.param_groups[0]["params"]
        assert len(group) == len(old_params) + 6 and all(any(p is q for q in group) for p in new)
        assert all(p not in opt.state for p in new)
        # the group's order is a fresh child's module order
        assert [id(p) for p in group] == [id(p) for p in model.policy.parameters() if any(p is q for q in group)]
        _step(model, *rows)
    carried_step = float(opt.state[old_params[0]]["step"])
    for p in new:
        st = opt.state[p]
        assert tuple(st["exp_avg"].shape) == tuple(p.shape) == tuple(st["exp_avg_sq"].shape)
        assert float(st["step"]) == 1.0
    assert carried_step == 2.0
    # the zero output weight is the one branch tensor with a gradient at the fork (linear1's is zero through it)
    l2 = [p for n, p in model.policy.named_parameters() if n.startswith("features_extractor.") and "ffn_extra.linear2" in n]
    assert all(float(p.detach().abs().max()) > 0.0 for p in l2)


def test_w3_256_builds_nothing_and_a_1024_build_moves_no_other_initial_byte() -> None:
    with torch_globals(num_threads=1):
        base = _toy().policy.state_dict()
        wide = _toy(trunk_ffn_dim=1024).policy.state_dict()
    assert not any("ffn_extra" in k for k in base)
    assert [k for k in wide if "ffn_extra" not in k] == list(base)
    assert all(torch.equal(base[k], wide[k]) for k in base)


def test_w4_a_save_load_round_trip_keeps_the_function_and_records_the_width(rows, tmp_path) -> None:
    from agents.model.model_version import ModelVersion, ModelVersionError
    from agents.model.model_version.migrations import _migrate_config
    from agents.model.snapshot import (historical_load_kwargs, load_checkpoint_strict, load_model_snapshot,
                                       recorded_model_version, save_model_snapshot)

    with torch_globals(num_threads=1):
        model = _parent(rows)
        parent_version = recorded_model_version(model)
        G.apply_growth(model, FIELDS, rows=rows)
        ref = _fwd(model, rows)
        run = tmp_path / "child"
        run.mkdir()
        model.save(str(run / "final_model"))
        version = recorded_model_version(model)
        save_model_snapshot(str(run), version, git_hash="test")
        zp = str(run / "final_model.zip")
        bare = load_checkpoint_strict(zp, device="cpu", **historical_load_kwargs(zp))   # main.anchors' bare loader
        snap = load_model_snapshot(zp, env=None, current_version=version, device="cpu")
        got_bare, got_snap = _fwd(bare, rows), _fwd(snap, rows)
        saved = ModelVersion.from_json_file(str(run / "model_config.json"))
    assert parent_version.trunk_ffn_dim == 256 and version.trunk_ffn_dim == 1024 and saved.trunk_ffn_dim == 1024
    assert _same(ref, got_bare) and _same(ref, got_snap)
    assert G.trunk_ffn_dim_of(bare.policy) == 1024 and G.trunk_ffn_dim_of(snap.policy) == 1024
    with pytest.raises(ModelVersionError, match="(?s)trunk_ffn_dim mismatch.*--grow trunk_ffn_dim=N"):
        parent_version.check_compatible(saved)
    # a pre-v164 config (no field) migrates to the only width ever built
    raw = dataclasses.asdict(parent_version)
    raw.pop("trunk_ffn_dim")
    raw["config_version"] = 163
    assert _migrate_config(raw)["trunk_ffn_dim"] == 256


def test_w5_depth_growth_on_a_wide_parent_and_depth_plus_width_at_one_fork(rows) -> None:
    with torch_globals(num_threads=1):
        wide = _parent(rows, trunk_ffn_dim=1024)
        before = _fwd(wide, rows)
        trained = [p for n, p in wide.policy.named_parameters() if ".ffn_extra." in n]
        r1 = G.apply_growth(wide, {"trunk_layers": {"parent": 2, "child": 3}}, rows=rows)
        assert r1["function_check"]["bit_identical"] and _same(before, _fwd(wide, rows))
        assert list(wide.policy.state_dict()) == list(_toy(3, 1024).policy.state_dict())
        rnd = wide.policy.features_extractor.team_transformer.extra_rounds[0]
        assert rnd.ffn_extra is not None and float(rnd.ffn_extra.linear2.weight.detach().abs().max()) == 0.0
        assert all(any(p is q for q in wide.policy.parameters()) for p in trained)    # the trained branches kept

        both = _parent(rows)
        before = _fwd(both, rows)
        r2 = G.apply_growth(both, {"trunk_layers": {"parent": 2, "child": 3},
                                   "trunk_ffn_dim": {"parent": 256, "child": 1024}}, rows=rows)
        assert r2["function_check"]["bit_identical"] and _same(before, _fwd(both, rows))
        fresh = _toy(3, 1024).policy.state_dict()
    sd = both.policy.state_dict()
    assert list(sd) == list(fresh)
    assert all(torch.equal(sd[k], fresh[k]) for k in sd if ".ffn_extra." in k)


def test_w6_the_refusals() -> None:
    from agents.model.trunk_ffn_width import check_ffn_dim

    assert G.parse_value("trunk_ffn_dim", "1024") == 1024
    for bad in ("128", "wide", "1024.5"):
        with pytest.raises(G.GrowthError):
            G.parse_value("trunk_ffn_dim", bad)
    with pytest.raises(ValueError):
        check_ffn_dim(255.0)
    with pytest.raises(G.GrowthError, match="only WIDENS"):
        G.validate_step("trunk_ffn_dim", 1024, 512)
    with pytest.raises(G.GrowthError, match="already 512 wide"):
        G.validate_step("trunk_ffn_dim", 512, 1024)
    G.validate_step("trunk_ffn_dim", 256, 1024)
    # the per-kind expert pair is refused where the trunk is built
    from agents.model.team_transformer import TeamTransformer
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    with pytest.raises(ValueError, match="refused beside trunk_ffn_by_kind"):
        TeamTransformer(layout, token_encoding="static", trunk_ffn_by_kind="on", trunk_ffn_dim=512)
    assert TeamTransformer(layout, token_encoding="static", trunk_ffn_dim=512).transformer_layers[1].ffn_extra is not None


@pytest.mark.slow  # two CPU Inductor compiles of the toy extractor (the growth_test compile test's pattern, ~80 s)
def test_w7_under_compile_the_widened_toy_matches_the_parent_within_compile_noise(rows) -> None:
    """The width branch compiles (CPU Inductor, through `compile_on`) and the compiled grown toy sits within compile
    noise of the compiled AND the eager parent on the gate's 64 real rows."""
    from agents.model.growth_test import COMPILE_ATOL
    from agents.model.inductor_compile import compile_on

    with torch_globals(num_threads=1):
        try:
            model = _parent(rows)
            parent = G.cpu_probe(model)
            eager_parent = G.forward_record(parent, *rows)
            parent.features_extractor.forward = compile_on(parent.features_extractor.forward, device="cpu",
                                                           dynamic=False)
            compiled_parent = G.forward_record(parent, *rows)
            G.apply_growth(model, FIELDS, rows=rows)
            child = G.cpu_probe(model)
            child.features_extractor.forward = compile_on(child.features_extractor.forward, device="cpu",
                                                          dynamic=False)
            compiled_child = G.forward_record(child, *rows)
        finally:
            torch._dynamo.reset()
    for ref in (compiled_parent, eager_parent):
        fin = np.isfinite(ref["logits"])
        assert np.array_equal(fin, np.isfinite(compiled_child["logits"]))
        d = max(float(np.abs(compiled_child["logits"][fin] - ref["logits"][fin]).max()),
                float(np.abs(compiled_child["values"] - ref["values"]).max()))
        assert d <= COMPILE_ATOL, f"the compiled widened toy is {d:.3g} from the parent (bar {COMPILE_ATOL})"
