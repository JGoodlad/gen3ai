"""K9(b)'s declared discrete-site inventory (`selection_sites`, `gen3_behaviour_tie_exclusion_v1`).

STATIC: every selection / value-position comparison / float -> int cast in a forward module is declared,
no declaration is stale, no line mixes classes, and a planted new topk / comparison is caught.
RUNTIME (the production forward on the learner golden's real rows): every torch op of the forward is
issued from a declared forward module; no discrete op on a float operand runs at an undeclared line; and
an EXACT site's operands do not move under a few-ulp weight jitter (a SCORE declared exact would)."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch as th
from torch.overrides import TorchFunctionMode

from agents.model import selection_sites as SS


def _undeclared(module: str, nodes: List[SS.Node]) -> List[str]:
    return [f"{module}.py:{n.line} {n.kind} {n.src!r}" for n in nodes
            if SS.declaration(module, n.src, n.func) is None]


def test_every_discrete_node_of_every_forward_module_is_declared() -> None:
    bad: List[str] = []
    for m in SS.FORWARD_MODULES:
        bad += _undeclared(m, SS.scan(m))
    assert not bad, ("UNDECLARED discrete op(s) in the policy forward — declare each in "
                     "agents/model/selection_sites.py (MARGIN rule for a score, EXACT reason otherwise):\n  "
                     + "\n  ".join(bad))


def test_no_declaration_is_stale_and_every_rule_is_known() -> None:
    present = {(m, n.src) for m in SS.FORWARD_MODULES for n in SS.scan(m)}
    stale = [k for k in SS.MARGIN if k not in present]
    stale += [(m, src) for m, by in SS.EXACT.items() for srcs in by.values() for src in srcs if (m, src) not in present]
    assert not stale, f"declarations naming source that no longer exists: {stale}"
    assert all(m in SS.FORWARD_MODULES for m in SS.EXACT)
    assert all(r.kind in SS.RULE_KINDS for r in SS.MARGIN.values())
    assert all(reason in SS.REASONS for by in SS.EXACT.values() for reason in by)
    kinds = {(m, n.src): n.kind for m in SS.FORWARD_MODULES for n in SS.scan(m)}
    assert all(kinds[k] != "cast" for k in SS.MARGIN), "no margin rule exists for a cast"
    dup = [(m, src) for m, by in SS.EXACT.items() for srcs in by.values() for src in srcs if (m, src) in SS.MARGIN]
    assert not dup, f"declared both MARGIN and EXACT: {dup}"


def _root_name(n: Any) -> str:
    import ast
    while isinstance(n, (ast.Subscript, ast.Attribute, ast.Call)):
        n = n.value if not isinstance(n, ast.Call) else n.func
    return n.id if isinstance(n, ast.Name) else ""


def _payload_violations(source: str, sel_src: str, payload: Tuple[str, ...]) -> List[str]:
    """Every use of the selection's index other than as the INDEX of a ``torch.gather`` over a declared
    payload (`gen3_behaviour_tie_identity_v1`: an undeclared consumer would make payload identity unsound)."""
    import ast
    tree = ast.parse(source)
    out: List[str] = []
    found = 0
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        direct = [a for a in fn.body for a in ast.walk(a)]
        for a in direct:
            if not (isinstance(a, ast.Assign) and ast.unparse(a.value) == sel_src):
                continue
            if any(isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)) and inner is not fn
                   and a in list(ast.walk(inner)) for inner in ast.walk(fn)):
                continue                      # belongs to a nested function: judged there
            found += 1
            if len(a.targets) != 1 or not isinstance(a.targets[0], ast.Name):
                out.append(f"{sel_src}: the index is not bound to one plain name")
                continue
            name = a.targets[0].id
            ok_ids = set()
            for c in ast.walk(fn):
                if (isinstance(c, ast.Call) and ast.unparse(c.func) in ("torch.gather", "th.gather")
                        and len(c.args) == 3 and isinstance(c.args[2], ast.Name) and c.args[2].id == name):
                    if _root_name(c.args[0]) in payload:
                        ok_ids.add(id(c.args[2]))
                    else:
                        out.append(f"{sel_src}: gathers {ast.unparse(c.args[0])!r}, not a declared payload {payload}")
            for u in ast.walk(fn):
                if (isinstance(u, ast.Name) and u.id == name and isinstance(u.ctx, ast.Load)
                        and id(u) not in ok_ids):
                    out.append(f"{sel_src}: its index {name!r} is read outside a declared payload gather "
                               f"(line {u.lineno})")
    if found == 0:
        out.append(f"{sel_src}: no `<name> = {sel_src}` binding found")
    return out


def test_every_payload_sites_index_is_read_only_by_a_gather_of_its_declared_payload() -> None:
    """A payload rule (identity clearance) is sound only if the payload is EVERYTHING the index reaches."""
    rules = [(k, r) for k, r in SS.MARGIN.items() if r.payload]
    assert rules, "no payload rule is declared — the identity clearance would be vacuous"
    bad: List[str] = []
    for (module, src), r in rules:
        assert r.kind == "argmax", (module, src)
        bad += _payload_violations(SS.module_path(module).read_text(), src, r.payload)
    assert not bad, bad
    # teeth: a second consumer of the index, or a gather of an undeclared tensor, is caught
    planted = ("def f(wfc, acc, other):\n"
               "    dom = wfc.argmax(dim=-1, keepdim=True)\n"
               "    a = torch.gather(acc, -1, dom)\n"
               "    b = torch.gather(other, -1, dom)\n"
               "    return a + b + dom.float()\n")
    v = _payload_violations(planted, "wfc.argmax(dim=-1, keepdim=True)", ("acc",))
    assert any("not a declared payload" in x for x in v) and any("outside a declared payload" in x for x in v), v


def test_every_max_value_sites_index_is_read_only_by_a_gather_of_its_own_operand() -> None:
    """A MAX_VALUE site (gen3_fm_index_max_v1) is EXACT only because its index selects the operand's own
    maximum: any other consumer of the index — or a gather of a DIFFERENT tensor — would let a near-tie
    flip reach log pi, so the site would have to be a MARGIN site."""
    import ast
    sites = [(m, src) for m, by in SS.EXACT.items() for src in by.get("MAX_VALUE", ())]
    assert sites, "no MAX_VALUE site is declared — this check would be vacuous"
    bad: List[str] = []
    for module, src in sites:
        call = ast.parse(src, mode="eval").body
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) \
            and call.func.attr == "argmax", (module, src)
        operand = _root_name(call.func.value)
        bad += _payload_violations(SS.module_path(module).read_text(), src, (operand,))
    assert not bad, bad
    # teeth: the same index gathering ANOTHER tensor is caught
    planted = ("def f(x, y):\n"
               "    idx = x.detach().argmax(dim=-1, keepdim=True)\n"
               "    return torch.gather(y, -1, idx).squeeze(-1)\n")
    assert _payload_violations(planted, "x.detach().argmax(dim=-1, keepdim=True)", ("x",))


def test_no_line_mixes_a_margin_and_an_exact_op_of_one_kind() -> None:
    amb = [x for m in SS.FORWARD_MODULES for x in SS.ambiguous_lines(m)]
    assert not amb, f"the recorder resolves ops by LINE — split these lines: {amb}"


def test_a_planted_topk_argmax_or_score_comparison_is_caught() -> None:
    src = ("def f(x, y, k):\n"
           "    a = x.topk(k, dim=-1)\n"
           "    b = y.argmax(dim=-1)\n"
           "    c = (x > y).float()\n"
           "    d = torch.ge(x, 0.5)\n"
           "    e = x.to(torch.long)\n"
           "    if k > 0:\n"                  # a Python test, not a tensor op: not scanned
           "        pass\n"
           "    return a, b, c, d, e\n")
    nodes = SS.scan_source("damage_op", src)
    assert sorted((n.kind, n.src) for n in nodes) == sorted([
        ("sel", "x.topk(k, dim=-1)"), ("sel", "y.argmax(dim=-1)"), ("cmp", "x > y"),
        ("cmp", "torch.ge(x, 0.5)"), ("cast", "x.to(torch.long)")])
    assert len(_undeclared("damage_op", nodes)) == 5


# ----------------------------------------------------------------------------------------- runtime
class _Ops(TorchFunctionMode):
    """Every torch op's issuing file, and every discrete op's float operands (by site, in order)."""

    def __init__(self) -> None:
        super().__init__()
        self.files: Dict[str, int] = {}
        self.events: List[Tuple[str, int, str, List[th.Tensor]]] = []

    def __torch_function__(self, func: Any, types: Any, args: Tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        f = sys._getframe(1)
        torch_dir = os.path.dirname(os.path.abspath(th.__file__))
        while f is not None and f.f_code.co_filename.startswith(torch_dir):
            f = f.f_back
        if f is None:
            return out
        fn = f.f_code.co_filename
        if "/agents/model/" in fn:
            mod = os.path.basename(fn)[:-3]
            self.files[mod] = self.files.get(mod, 0) + 1
            r = SS.runtime_op(getattr(func, "__name__", ""))
            if r is not None:
                fl = [a.detach().clone() for a in args[:2] if isinstance(a, th.Tensor) and a.is_floating_point()]
                if fl:
                    self.events.append((mod, int(f.f_lineno), r[0], fl))
        return out


@pytest.fixture(scope="module")
def production() -> Tuple[Any, Dict[str, th.Tensor], th.Tensor, th.Tensor]:
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.training import learner_golden as LG

    th.manual_seed(0)
    model = LG.build_learner()
    LG.load_buffer_into(model)
    rb = model.rollout_buffer
    obs = obs_as_tensor({k: v.reshape((-1,) + v.shape[2:]) for k, v in rb.observations.items()}, "cpu")
    acts = th.as_tensor(rb.actions.reshape(-1)).long()
    masks = th.as_tensor(rb.action_masks.reshape(acts.shape[0], -1))
    return model, obs, acts, masks


def _forward(model: Any, obs: Any, acts: th.Tensor, masks: th.Tensor) -> _Ops:
    rec = _Ops()
    model.policy.set_training_mode(True)
    with th.no_grad(), rec:
        model.policy.evaluate_actions(obs, acts, action_masks=masks)
    return rec


def test_the_production_forward_runs_only_declared_modules_and_sites(production: Any) -> None:
    from agents.training.rust_rollout.tie_margins import TieMargins

    model, obs, acts, masks = production
    rec = _forward(model, obs, acts, masks)
    outside = sorted(set(rec.files) - set(SS.FORWARD_MODULES))
    assert not outside, f"the forward runs torch ops from modules outside FORWARD_MODULES: {outside}"
    tm = TieMargins(int(acts.shape[0]))
    with th.no_grad(), tm:
        model.policy.evaluate_actions(obs, acts, action_masks=masks)
    tm.check()                                       # no undeclared discrete op on a float operand
    assert tm.sites_seen, "the production forward executed no MARGIN site — the recorder is not attached"
    assert np.isfinite(tm.margin).all() and (tm.margin >= 0).all()


def test_an_exact_site_does_not_move_under_weight_jitter(production: Any) -> None:
    """A SCORE declared EXACT would let a tie flip through un-excluded: its operands move with the weights."""
    model, obs, acts, masks = production
    base = _forward(model, obs, acts, masks).events
    saved = {k: p.detach().clone() for k, p in model.policy.named_parameters()}
    g = th.Generator().manual_seed(1)
    try:
        with th.no_grad():
            for p in model.policy.parameters():
                if p.is_floating_point():
                    p.mul_(1 + 3e-7 * th.randn(p.shape, generator=g))
        jit = _forward(model, obs, acts, masks).events
    finally:
        with th.no_grad():
            for k, p in model.policy.named_parameters():
                p.copy_(saved[k])
    assert len(base) == len(jit)
    moved: Dict[str, str] = {}
    margin_moved = 0
    for (m, ln, kind, a), (m2, ln2, _k, b) in zip(base, jit):
        assert (m, ln) == (m2, ln2)
        res = SS.resolve(m, ln, kind)
        if res is None or res.declared is None:
            continue
        if any(x.shape != y.shape or not th.equal(x, y) for x, y in zip(a, b)):
            if res.declared.rule is None:
                moved[f"{m}.py:{ln}"] = res.src
            margin_moved += res.declared.rule is not None
    assert not moved, f"EXACT sites whose operands moved with the weights (declare them MARGIN): {moved}"
    assert margin_moved > 0, "no MARGIN site moved under a few-ulp weight jitter — the oracle saw nothing"
