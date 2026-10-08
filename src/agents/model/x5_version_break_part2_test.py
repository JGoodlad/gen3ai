"""THE X5 VERSION BREAK, part 2 — the EXACT-refactor bundle (config v144, `gen3_x5_version_break_v1`;
architecture audit F1 / F6a / F7a / F16b, `designs/endstate/design_arch_audit.md`, + the blob path's leftovers).

Each test FAILS ON REVERT of one piece:

* F1 — no value-tower parameter on the production policy or the T2-served module, the extractor's value half
  IS `value_pooled`, the critic value equals `sigmoid(win_head(value_pooled))`, and a non-winprob critic is
  refused at construction;
* F6a — no value-reduction `amax` / `amin` / `max(dim)` is left in a forward module outside a declared list of
  sites that sit on no gradient path; `max_by_index` routes an exact tie to the FIRST maximum with a finite
  gradient;
* F7a — the default (`--speed-physics off`) production forward reads no speed-spread sigma (an instrumented
  `SPECIES_SPREAD_PRIOR`);
* F16b — the flat opponent pointer's shared scorer has no bias;
* the blob leftovers — `BeliefSlots` / `AlphaIntentHead` / `BetaSwitchHead`, the alpha / beta stashes and
  `--beta-setvalued-coef` are gone (a typed flag is refused with its reason), and a RETIRED module's keys are
  refused by a strict load (the "registered as None" hole).

The weight-mapping identity proof is a committed MEASUREMENT, not a test (it needs the pre-break reference
weights, which are not committed): `designs/research_state/measurements/version_break_identity_2026-10-07/`.
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Set, Tuple

import pytest
import torch


@pytest.fixture(scope="module")
def learner():
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    LG.load_buffer_into(m, LG.BUFFER_PATH)
    return m


def _rows(m: Any, n: int = 16) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    rb = m.rollout_buffer
    obs = {k: torch.as_tensor(v.reshape(-1, *v.shape[2:]))[:n] for k, v in rb.observations.items()}
    masks = torch.as_tensor(rb.action_masks.reshape(-1, rb.action_masks.shape[-1]))[:n]
    return obs, masks


_TOWER = ("value_pre_norm", "value_projection", "value_net")


# ----------------------------------------------------------------------------------------------------- F1
def test_the_production_policy_holds_no_value_tower(learner):
    pol = learner.policy
    keys = list(pol.state_dict())
    for prefix in ("features_extractor.value_pre_norm.", "features_extractor.value_projection.",
                   "mlp_extractor.value_net.", "value_net."):
        assert not [k for k in keys if k.startswith(prefix)], prefix
    assert not hasattr(pol.features_extractor, "value_projection")
    assert not list(pol.mlp_extractor.value_net.parameters())
    assert not list(pol.value_net.parameters()) and not list(pol.action_net.parameters())
    with pytest.raises(RuntimeError, match="value_net was deleted"):
        pol.value_net(torch.zeros(1, 128))
    # every parameter is in the optimizer, and nothing else is
    in_opt = {id(p) for g in pol.optimizer.param_groups for p in g["params"]}
    assert in_opt == {id(p) for p in pol.parameters()}


def test_the_T2_served_module_holds_no_value_tower_and_serves_the_win_head(learner):
    from agents.inference.service.decision import DecisionModule
    pol = learner.policy
    dm = DecisionModule(pol).eval()
    names = [n for n, _ in dm.named_parameters()]
    assert names and not [n for n in names if any(t in n for t in _TOWER)], names
    obs, masks = _rows(learner)
    with torch.no_grad():
        _logp, value = dm(obs["observation"], masks.bool())
        fe = pol.features_extractor
        expect = torch.sigmoid(fe.win_head(fe.last_value_pooled)).reshape(-1)
    assert torch.equal(value, expect)


def test_the_value_half_is_value_pooled_and_the_critic_is_sigmoid_of_the_win_head(learner):
    pol = learner.policy
    obs, _masks = _rows(learner)
    with torch.no_grad():
        _pi, vf = pol.extract_features(obs)
        fe = pol.features_extractor
        assert vf is fe.last_value_pooled
        assert vf.shape[-1] == fe.vf_features_dim
        v = pol.predict_values(obs)
        expect = torch.sigmoid(fe.win_head(fe.last_value_pooled))
    assert v.shape == expect.shape and torch.equal(v, expect)


def test_a_non_winprob_critic_is_refused_at_construction(learner):
    pol = learner.policy
    params = dict(pol._get_constructor_parameters())
    params["critic"] = "shaped"
    with pytest.raises(ValueError, match="value_net a 'shaped' critic read was DELETED"):
        type(pol)(**params)


# ---------------------------------------------------------------------------------------------------- F6a
#: The value-reduction max sites that STAY `amax` / `amin` / `max(dim)`, each on NO gradient path. A new
#: entry needs that reason; a reduction on a gradient path is `index_max.max_by_index`.
_NON_GRADIENT_MAXES: Dict[Tuple[str, str], str] = {
    ("damage_op", "wh.amax(dim=-1)"): "the provenance gate's operand: a comparison (> eps), no gradient",
    ("damage_op_blocks", "self.MOVE_CURES_SELF_STATUS[our_moves].amax(dim=-1)"): "a constant TABLE gathered "
                                                                                 "at integer move ids",
    ("damage_op_blocks", "self.MOVE_CURES_TEAM_STATUS[our_moves].amax(dim=-1)"): "a constant TABLE gathered "
                                                                                 "at integer move ids",
    ("damage_op_pairwise", "((ctx.all_move_ids[:, :TEAM_SIZE] == pur).any(-1).float() * alive_i)"
                           ".amax(dim=-1, keepdim=True)"): "an observation indicator x an observation mask",
    ("damage_op_blocks", "paths.amin(dim=-1)"): "the cheapest status-undo path: constants selected by "
                                                "observation / table masks",
    ("hypothesis_set", "gaps.amin(-1)"): "boundary_gap: the move-order TIE-MARGIN diagnostic (K9(b)'s near-tie "
                                         "read), never in a loss",
    ("hypothesis_tokens", "torch.where(ctx.opp_addressable, gap, torch.full_like(gap, float('inf'))).amin(-1)"):
        "the roster's move-tie-gap DIAGNOSTIC (K9(b)'s near-tie read), never in a loss",
    ("hypothesis_set", "torch.where(cand, s, torch.full_like(s, -math.inf)).amax(-1)"): "fixed_size_tau's "
                                                                                       "bracket, under no_grad",
    ("hypothesis_set", "torch.where(cand, s, torch.full_like(s, math.inf)).amin(-1)"): "fixed_size_tau's "
                                                                                      "bracket, under no_grad",
    ("opp_intent", "p_species.max(dim=-1)"): "resolve_believed_slot_by_content: a LABEL helper, not the forward",
}


def _value_maxes(module: str) -> Set[str]:
    from agents.model import selection_sites as SS
    tree = ast.parse(SS.module_path(module).read_text())
    out: Set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        name = node.func.attr
        is_torch_fn = isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
        if name in ("amax", "amin"):
            out.add(ast.unparse(node))
        elif name in ("max", "min"):
            # a DIM reduction: `x.max(dim=...)`, `x.max(<int>)`, `torch.max(x, <dim>)`. Not `x.max()` (a
            # scalar read) and not `x.max(other)` / `torch.max(x)` (elementwise / full).
            n_pos = len(node.args) - (1 if is_torch_fn else 0)
            has_dim = any(k.arg == "dim" for k in node.keywords) or (
                n_pos == 1 and isinstance(node.args[-1], ast.Constant) and isinstance(node.args[-1].value, int))
            if has_dim:
                out.add(ast.unparse(node))
    return out


def test_no_forward_module_reduces_a_value_with_amax_outside_the_declared_non_gradient_sites():
    from agents.model import selection_sites as SS
    found = {(m, src) for m in SS.FORWARD_MODULES if m != "index_max" for src in _value_maxes(m)}
    undeclared = sorted(found - set(_NON_GRADIENT_MAXES))
    assert not undeclared, (
        "a value-reduction max outside `index_max.max_by_index` (audit F6a: amax's backward divides by a float "
        f"tie count a compiled recompute can zero — NaN). Use max_by_index, or declare it here: {undeclared}")
    stale = sorted(set(_NON_GRADIENT_MAXES) - found)
    assert not stale, f"declared non-gradient max sites that no longer exist: {stale}"


def test_the_scan_catches_a_planted_amax():
    src = "def f(x, w):\n    a = (w * x).amax(dim=-1)\n    b = x.max(dim=1).values\n    return a + b + x.max()\n"
    tree = ast.parse(src)
    hits = [ast.unparse(n) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("amax", "max")
            and (n.args or n.keywords)]
    assert "(w * x).amax(dim=-1)" in hits and "x.max(dim=1)" in hits and "x.max()" not in hits


@pytest.mark.parametrize("dim,keepdim", [(-1, False), (1, False), (1, True)])
def test_max_by_index_is_amax_in_value_and_routes_an_exact_tie_to_the_first_max(dim, keepdim):
    from agents.model.index_max import max_by_index
    x = torch.tensor([[[0.0, 2.0, 2.0], [1.0, 1.0, 0.5]], [[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]],
                     requires_grad=True)
    v = max_by_index(x, dim=dim, keepdim=keepdim)
    assert torch.equal(v, x.amax(dim=dim, keepdim=keepdim))
    v.sum().backward()
    g = x.grad
    assert g is not None and bool(torch.isfinite(g).all())
    # every reduced slice sends its whole unit of gradient to exactly ONE element: the first maximum
    idx = x.detach().argmax(dim=dim, keepdim=True)
    expect = torch.zeros_like(x).scatter_(dim, idx, 1.0)
    assert torch.equal(g, expect)


# ---------------------------------------------------------------------------------------------------- F7a
class _SigmaSpy(torch.Tensor):
    """A `SPECIES_SPREAD_PRIOR` that RECORDS every read of the speed-spread sigma column (`[..., spe, 1]`)."""
    reads: List[str] = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        if func is torch.Tensor.__getitem__ and isinstance(args[0], _SigmaSpy):
            key = args[1]
            if isinstance(key, tuple) and key and isinstance(key[-1], int) and key[-1] == 1 and len(key) >= 2:
                cls.reads.append(repr(tuple(type(k).__name__ for k in key)))
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))


def test_the_default_forward_reads_no_speed_spread_sigma(learner):
    pol = learner.policy
    op = pol.features_extractor.damage_op
    assert op is not None and not op.speed_physics, "PRECONDITION: the production op, --speed-physics off"
    real = op._buffers["SPECIES_SPREAD_PRIOR"]
    _SigmaSpy.reads = []
    op._buffers["SPECIES_SPREAD_PRIOR"] = real.as_subclass(_SigmaSpy)
    try:
        # teeth: the instrument sees a sigma read
        _ = op.SPECIES_SPREAD_PRIOR[torch.tensor([1]), 4, 1]
        assert _SigmaSpy.reads, "the spy saw no read — this test would be vacuous"
        _SigmaSpy.reads = []
        obs, masks = _rows(learner, 8)
        with torch.no_grad():
            pol.evaluate_actions(obs, torch.zeros(8, dtype=torch.long), action_masks=masks)
    finally:
        op._buffers["SPECIES_SPREAD_PRIOR"] = real
    assert not _SigmaSpy.reads, f"the default forward read the speed-spread sigma {len(_SigmaSpy.reads)}x"


def test_the_speed_physics_forward_reads_no_speed_spread_sigma_either():
    """F7a COMPLETED (the version break's part 5, after `gen3_speed_mixture_v1`): `--speed-physics on` reads the
    discrete Smogon speed mixture, so the X5 OTHER roster's speed-SPREAD average (`OpRoster.spe_std`, built only
    under `on`) had no reader in either mode — it is deleted, and an `on` forward reads no sigma at all."""
    import dataclasses
    import inspect
    import json

    import gymnasium as gym
    import numpy as np

    from agents.model.damage_tables import sanitize_historical_move_floor
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.hypothesis_tokens import OpRoster, other_roster
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from utils.paths import repo_path, src_path
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    kw = {k: v for k, v in cfg.items() if k in set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)}
    sanitize_historical_move_floor(kw)
    kw["speed_physics"] = "on"
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    torch.manual_seed(0)
    fe = Gen3FeaturesExtractor(gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32),
                               layout=layout, mappings=mappings, **kw).eval()
    op = fe.damage_op
    assert op.speed_physics and fe.hypothesis_builder is not None, "PRECONDITION: `on`, the X5 OTHER roster built"
    obs = torch.as_tensor(np.load(src_path("agents", "model", "compile_parity_obs.npz"))["obs"])[:16]
    real = op._buffers["SPECIES_SPREAD_PRIOR"]
    _SigmaSpy.reads = []
    op._buffers["SPECIES_SPREAD_PRIOR"] = real.as_subclass(_SigmaSpy)
    try:
        _ = op.SPECIES_SPREAD_PRIOR[:, 4, 1]                      # teeth: the OTHER roster's spelling is seen
        assert _SigmaSpy.reads, "the spy saw no read — this test would be vacuous"
        _SigmaSpy.reads = []
        with torch.no_grad():
            fe({"observation": obs})
    finally:
        op._buffers["SPECIES_SPREAD_PRIOR"] = real
    assert not _SigmaSpy.reads, f"the `on` forward read the speed-spread sigma {len(_SigmaSpy.reads)}x"
    assert "spe_std" not in {f.name for f in dataclasses.fields(OpRoster)}
    assert "with_spe_std" not in inspect.signature(other_roster).parameters


# --------------------------------------------------------------------------------------------------- F16b
def test_the_flat_pointer_scorer_has_no_bias(learner):
    fe = learner.policy.features_extractor
    assert fe.flat_intent_head is not None
    assert fe.flat_intent_head.out.bias is None
    assert "features_extractor.flat_intent_head.out.bias" not in learner.policy.state_dict()
    # the policy pointer head's three scorers carry PER-FAMILY biases inside one softmax (move vs switch vs
    # struggle) — not common to every logit, so not dead: they stay
    ph = learner.policy.pointer_head
    assert all(s.bias is not None for s in (ph.move_score, ph.switch_score, ph.struggle_score))


# ------------------------------------------------------------------------------------ the blob leftovers
def test_the_blob_modules_and_stashes_are_deleted(learner):
    from agents.model import belief_heads, extractor_api, opp_intent
    from agents.model.extractor_stashes import ExtractorStashes
    import dataclasses
    assert not hasattr(opp_intent, "AlphaIntentHead") and not hasattr(opp_intent, "BetaSwitchHead")
    assert not hasattr(belief_heads, "BeliefSlots")
    fields = {f.name for f in dataclasses.fields(ExtractorStashes)}
    assert not {"alpha_logits", "beta_logits", "alpha_seat_nums"} & fields
    for name in ("last_alpha_logits", "last_beta_logits", "last_alpha_seat_nums",
                 "retire_superseded_intent_heads"):
        assert not hasattr(extractor_api.ExtractorApi, name), name
    fe = learner.policy.features_extractor
    for name in ("alpha_head", "beta_head", "belief_slots"):
        assert not hasattr(fe, name), name
    assert "alpha_logits" not in belief_heads._BELIEF_SUPERVISION_KEYS


def test_a_typed_beta_setvalued_coef_is_refused_at_parse_time_with_its_reason(capsys):
    from main.train.parser import build_parser
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--beta-setvalued-coef", "0.05"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "--beta-setvalued-coef was DELETED" in err and "VERSION BREAK part 2" in err, err[-600:]


def test_the_production_recipe_carries_no_beta_setvalued_coef():
    from main.train import recipe_surface as rs
    from main.train.recipe_surface import production_recipe
    assert "beta_setvalued_coef" not in production_recipe()
    assert "beta_setvalued_coef" not in {r.dest for r in rs.ROWS}


def test_a_retired_modules_keys_are_refused_by_a_strict_load():
    """`drop_child` leaves a PLAIN None: torch then reports a key under the retired name as UNEXPECTED. The
    old `owner.child = None` kept `child` in `_modules` and a strict load SWALLOWED such a key."""
    from agents.model.extractor_api import drop_child
    owner = torch.nn.Module()
    owner.child = torch.nn.Linear(2, 2)
    owner.keep = torch.nn.Linear(2, 2)
    stale = owner.state_dict()
    owner.child = None                                     # the old retire: the hole
    owner.load_state_dict(stale, strict=True)             # swallowed (no error) — the defect, reproduced
    owner2 = torch.nn.Module()
    owner2.child = torch.nn.Linear(2, 2)
    owner2.keep = torch.nn.Linear(2, 2)
    drop_child(owner2, "child")
    assert owner2.child is None and "child" not in owner2._modules
    with pytest.raises(RuntimeError, match="Unexpected key"):
        owner2.load_state_dict(stale, strict=True)


def test_every_retire_hook_drops_its_child_through_drop_child():
    """The three retire hooks of `ExtractorApi` are spelled with `drop_child` (an AST read: a bare
    `x.name = None` on a module would reopen the hole)."""
    import inspect

    from agents.model import extractor_api
    tree = ast.parse(inspect.getsource(extractor_api.ExtractorApi))
    bad = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("retire_"):
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "drop_child"]
            nones = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Constant) and n.value.value is None]
            if not calls or nones:
                bad.append((fn.name, nones))
    assert not bad, bad
