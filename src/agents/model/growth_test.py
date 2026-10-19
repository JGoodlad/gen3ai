"""B-GROW (`agents.model.growth`, `gen3_grow_fork_v1`): a fork's declared trunk growth starts as its PARENT'S
FUNCTION, bit for bit, and its optimizer carries the parent's AdamW state while the new parameters start fresh.

The parent is the toy network (`adopted_levers_test.TOY_ARGV`, the X34 / X47 toy, depth 2) after ONE real AdamW step,
so every parameter holds non-trivial moments; the growth is X47's, 2 -> 4. Everything is built through SB3
(`build_fresh_model`, the production construction path), on the CPU, one torch thread, fp32 eager.

Each test fails on revert: without `grow_trunk_layers` the child has no `extra_rounds` keys and the fresh depth-4
build's strict load fails; without `regrow_optimizer` the new rounds are in no param group; a non-identity init of a
new round (the monkeypatched class below) is refused by the fork-time gate.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from agents.model import growth as G
from agents.model.model_version import ModelVersionError
from utils.torch_state_guard import torch_globals

FIELDS = {"trunk_layers": {"parent": 2, "child": 4}}
#: Two IdentityInitRounds at d_model 128, FFN 256 (the design doc's figure, `design_bot_ideas_screen.md` §2.1).
NEW_ELEMENTS = 264_960


def _toy(trunk_layers: int = 2) -> Any:
    from agents.model.adopted_levers_test import _toy_args
    from main.fresh_checkpoint import build_fresh_model

    args = copy.deepcopy(_toy_args())
    args.trunk_layers = trunk_layers
    model, _args, _pk = build_fresh_model(0, args=args)
    return model


def _rows() -> Tuple[np.ndarray, np.ndarray]:
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    return load_parity_rows(Gen3ObservationEncoder(load_mappings()).dimension)


def _step(model: Any, obs: np.ndarray, mask: np.ndarray) -> None:
    """One real AdamW step on a loss that reaches the policy and the critic (non-zero moments everywhere it reads)."""
    pol = model.policy
    pol.set_training_mode(True)
    o = {"observation": torch.as_tensor(obs[:16]), "action_mask": torch.as_tensor(mask[:16].astype(np.int8))}
    dist = pol.get_distribution(o, action_masks=mask[:16]).distribution
    loss = (dist.probs * torch.arange(dist.probs.shape[-1], dtype=torch.float32)).sum() + pol.predict_values(o).sum()
    pol.optimizer.zero_grad()
    loss.backward()
    pol.optimizer.step()
    pol.optimizer.zero_grad()
    pol.set_training_mode(False)


def _parent() -> Tuple[Any, np.ndarray, np.ndarray]:
    obs, mask = _rows()
    model = _toy(2)
    _step(model, obs, mask)
    return model, obs, mask


def _live_forward(model: Any, obs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    model.policy.set_training_mode(False)
    return G.forward_record(model.policy, obs, mask)


def test_the_grown_toy_is_its_parents_function_bit_for_bit_and_a_fresh_depth_4_build() -> None:
    with torch_globals(num_threads=1):
        model, obs, mask = _parent()
        before = _live_forward(model, obs, mask)
        keys_before = list(model.policy.state_dict())
        report = G.apply_growth(model, FIELDS, rows=(obs, mask))
        after = _live_forward(model, obs, mask)
        fresh4 = _toy(4)
    fc = report["function_check"]
    assert fc["bit_identical"] and fc["max_abs_d_logits"] == 0.0 and fc["max_abs_d_value"] == 0.0 and fc["rows"] == 64
    # the LIVE grown policy too (not only the gate's CPU builds)
    assert np.array_equal(before["logits"], after["logits"]) and np.array_equal(before["values"], after["values"])
    # the grown state dict IS a fresh depth-4 build's: same keys, same ORDER, same shapes
    sd, sd4 = model.policy.state_dict(), fresh4.policy.state_dict()
    assert list(sd) == list(sd4)
    assert all(tuple(sd[k].shape) == tuple(sd4[k].shape) for k in sd)
    new = [k for k in sd if k not in keys_before]
    assert new and all(".team_transformer.extra_rounds." in k for k in new)
    assert {k.split(".extra_rounds.")[1].split(".")[0] for k in new} == {"0", "1"}
    zero = [k for k in new if k.startswith("features_extractor.") and (".out_proj." in k or ".linear2." in k)]
    assert len(zero) == 8 and all(float(sd[k].abs().max()) == 0.0 for k in zero)
    assert report["new_parameters"] == {"tensors": 24, "elements": NEW_ELEMENTS}
    fe = model.policy.features_extractor
    assert fe.trunk_layers == 4 and fe.team_transformer.trunk_layers == 4
    assert model.policy_kwargs["features_extractor_kwargs"]["trunk_layers"] == 4
    assert model.policy.features_extractor_kwargs["trunk_layers"] == 4


def test_a_parent_that_already_has_an_extra_round_keeps_it_and_grows_after_it() -> None:
    """X47b's shape (production 3 -> 4), on the toy: the parent's TRAINED extra round stays round 0 (the same parameter
    objects, untouched) and the new one is appended AFTER it — an insertion before it would reorder the trunk."""
    obs, mask = _rows()
    with torch_globals(num_threads=1):
        model = _toy(3)
        _step(model, obs, mask)
        before = _live_forward(model, obs, mask)
        round0 = list(model.policy.features_extractor.team_transformer.extra_rounds[0].parameters())
        report = G.apply_growth(model, {"trunk_layers": {"parent": 3, "child": 4}}, rows=(obs, mask))
        after = _live_forward(model, obs, mask)
        fresh4 = _toy(4)
    rounds = model.policy.features_extractor.team_transformer.extra_rounds
    assert len(rounds) == 2 and all(a is b for a, b in zip(rounds[0].parameters(), round0))
    assert float(rounds[0].out_proj.weight.detach().abs().max()) > 0.0       # the parent's round TRAINED (one step moved it)
    assert float(rounds[1].out_proj.weight.detach().abs().max()) == 0.0      # the new one is the identity
    assert report["function_check"]["bit_identical"] and report["new_parameters"]["elements"] == NEW_ELEMENTS // 2
    assert np.array_equal(before["logits"], after["logits"]) and np.array_equal(before["values"], after["values"])
    assert list(model.policy.state_dict()) == list(fresh4.policy.state_dict())


def test_the_optimizer_carries_the_parents_adamw_state_and_the_new_parameters_start_fresh() -> None:
    with torch_globals(num_threads=1):
        model, obs, mask = _parent()
        old_opt = model.policy.optimizer
        old_params = list(old_opt.param_groups[0]["params"])
        old_state = {id(p): old_opt.state[p] for p in old_params}
        old_group = {k: v for k, v in old_opt.param_groups[0].items() if k != "params"}
        report = G.apply_growth(model, FIELDS, rows=(obs, mask))
        opt = model.policy.optimizer
        fresh4 = _toy(4)
    assert opt is not old_opt and type(opt) is type(old_opt)
    assert {k: v for k, v in opt.param_groups[0].items() if k != "params"} == old_group
    # every parent parameter: the SAME state object (exp_avg, exp_avg_sq, step), untouched
    assert all(old_state[id(p)] for p in old_params), "the parent step left a parameter without AdamW state"
    assert all(opt.state[p] is old_state[id(p)] for p in old_params)
    assert report["optimizer_state"] == {"carried": len(old_params), "no_state": 0, "fresh": 24}
    # every new parameter: in the group, NO state (AdamW zero-inits it at its first step)
    names = {id(p): n for n, p in model.policy.named_parameters()}
    new = [p for p in opt.param_groups[0]["params"] if id(p) not in old_state]
    assert len(new) == 24 and all(".extra_rounds." in names[id(p)] for p in new)
    assert all(p not in opt.state or not opt.state[p] for p in new)
    # the group's ORDER is a fresh depth-4 build's (what a later restart's by-name remap reads)
    fresh_names = {id(p): n for n, p in fresh4.policy.named_parameters()}
    assert [names[id(p)] for p in opt.param_groups[0]["params"]] == \
        [fresh_names[id(p)] for p in fresh4.policy.optimizer.param_groups[0]["params"]]
    # and it trains: one more step gives the new parameters step 1, the carried ones step 2
    with torch_globals(num_threads=1):
        _step(model, obs, mask)
    assert all(float(opt.state[p]["step"]) == 1.0 for p in new)
    assert all(float(opt.state[p]["step"]) == 2.0 for p in old_params)


def test_the_fork_time_gate_refuses_a_non_identity_init(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.model import trunk_depth

    class _Leaky(trunk_depth.IdentityInitRound):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            with torch.no_grad():
                self.out_proj.weight.fill_(1e-3)            # NOT zero: the round now adds something

    with torch_globals(num_threads=1):
        model, obs, mask = _parent()
        monkeypatch.setattr(trunk_depth, "IdentityInitRound", _Leaky)
        with pytest.raises(G.GrowthError, match="NOT its parent's function"):
            G.apply_growth(model, FIELDS, rows=(obs, mask))


def test_declared_growth_refusals() -> None:
    with pytest.raises(G.GrowthError, match="B-KFFN"):
        G.require_built("trunk_ffn_by_kind")
    with pytest.raises(G.GrowthError, match="B-INT"):
        G.require_built("opp_intent_grad_mode")
    with pytest.raises(G.GrowthError, match="not a growable field"):
        G.require_built("damage_op")
    with pytest.raises(G.GrowthError, match="only ADDS"):
        G.validate_step("trunk_layers", 3, 2)
    with pytest.raises(G.GrowthError, match="must be one of"):
        G.parse_value("trunk_layers", "9")
    assert G.parse_value("trunk_layers", " 4 ") == 4
    assert issubclass(G.GrowthError, ModelVersionError)          # the trainer's FATAL_CONFIG path catches it
    with torch_globals(num_threads=1):
        model, obs, mask = _parent()
    with pytest.raises(G.GrowthError, match="loaded network has trunk_layers=2"):
        G.apply_growth(model, {"trunk_layers": {"parent": 3, "child": 4}}, rows=(obs, mask))


def test_check_compatible_grown_lets_exactly_the_declared_field_differ() -> None:
    from agents.model.snapshot import recorded_model_version

    with torch_globals(num_threads=1):
        parent = recorded_model_version(_toy(2))
    child = dataclasses.replace(parent, trunk_layers=4)
    with pytest.raises(ModelVersionError, match="(?s)trunk_layers mismatch.*--grow trunk_layers=N"):
        child.check_compatible(parent)
    child.check_compatible_grown(parent, FIELDS)                     # the declared step: accepted
    with pytest.raises(ModelVersionError, match="obs_facts mismatch"):  # any OTHER difference: still refused
        dataclasses.replace(child, obs_facts="v1" if parent.obs_facts != "v1" else "off").check_compatible_grown(
            parent, FIELDS)
    with pytest.raises(G.GrowthError, match="records trunk_layers=2"):
        child.check_compatible_grown(parent, {"trunk_layers": {"parent": 3, "child": 4}})
    with pytest.raises(G.GrowthError, match="this run builds trunk_layers=4"):
        child.check_compatible_grown(parent, {"trunk_layers": {"parent": 2, "child": 3}})


#: Compile noise on the CPU (Inductor reorders fp32 arithmetic): measured 2.4e-7 compiled vs eager on these rows; a
#: missing or non-identity round moves the logits by O(1e-3) or more (the gate test above).
COMPILE_ATOL = 1e-5


@pytest.mark.slow  # ~80 s: two CPU Inductor compiles of the toy extractor (measured 2026-10-19 on the agent cores)
def test_under_compile_the_grown_toy_matches_the_parent_within_compile_noise() -> None:
    """The compiled grown network against the compiled AND the eager parent, on the gate's 64 real rows (CPU Inductor;
    measured 2026-10-19: compiled child vs compiled parent max |d| 0.0, compiled vs eager 2.4e-7)."""
    with torch_globals(num_threads=1):
        try:
            model, obs, mask = _parent()
            parent = G.cpu_probe(model)
            eager_parent = G.forward_record(parent, obs, mask)
            parent.features_extractor.forward = torch.compile(parent.features_extractor.forward, dynamic=False)
            compiled_parent = G.forward_record(parent, obs, mask)
            G.apply_growth(model, FIELDS, rows=(obs, mask))
            child = G.cpu_probe(model)
            child.features_extractor.forward = torch.compile(child.features_extractor.forward, dynamic=False)
            compiled_child = G.forward_record(child, obs, mask)
        finally:
            torch._dynamo.reset()
    for ref in (compiled_parent, eager_parent):
        fin = np.isfinite(ref["logits"])
        assert np.array_equal(fin, np.isfinite(compiled_child["logits"]))
        d = max(float(np.abs(compiled_child["logits"][fin] - ref["logits"][fin]).max()),
                float(np.abs(compiled_child["values"] - ref["values"]).max()))
        assert d <= COMPILE_ATOL, f"the compiled grown toy is {d:.3g} from the parent (bar {COMPILE_ATOL})"
