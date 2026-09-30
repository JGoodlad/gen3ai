"""The eval declaration's FIXED-opponent slots (M5 Lane H, ``rust_eval.build``) — pure unit.

A T2 slot starts with its GROUP TEMPLATE's weights, and a same-architecture fixed opponent joins the
trainee's group, so a non-reused fixed slot plays the wrong policy until ``build_eval_core`` LOADS it
(found by the Lane H fixed-opponent gate row, ``parity_test.test_commit_fixed_opponents_…``). These pin
``load_fixed_slots``: every non-reused fixed slot is loaded with ITS policy, a reused one (the training
plan's stable slot) is left to its own host, a missing policy is refused.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.training.rust_eval.build import EvalDecl, eval_extra_slots, eval_table, load_fixed_slots


class _Svc:
    def __init__(self):
        self.loads = []

    def load(self, slot, policy, model_id):
        self.loads.append((slot, policy, model_id))


def test_every_non_reused_fixed_slot_is_loaded_with_its_own_policy():
    trainee, a, b, c = (SimpleNamespace(name=n) for n in ("trainee", "a", "b", "c"))
    decl = EvalDecl(n_envs=4, n_sentinels=2, fixed_labels=("ext_a", "ext_b", "ext_c"), reused_fixed=(("ext_b", 7),))
    pols = {"ext_a": a, "ext_b": b, "ext_c": c}
    extra = eval_extra_slots(decl, trainee, pols)
    assert [f for f, _p in extra] == ["evaltrainee", "evalsentinel", "evalsentinel", "evalfixed", "evalfixed"]
    ids = list(range(10, 10 + len(extra)))
    table = eval_table(decl, ids, ("heuristic",))
    assert table.fixed_slots == (("ext_a", 13), ("ext_b", 7), ("ext_c", 14))
    svc = _Svc()
    assert load_fixed_slots(svc, decl, table, pols) == [("ext_a", 13), ("ext_c", 14)]
    assert [(s, p.name, m) for s, p, m in svc.loads] == [(13, "a", "eval:fixed:ext_a"), (14, "c", "eval:fixed:ext_c")]


def test_no_fixed_opponent_loads_nothing_and_a_missing_policy_is_refused():
    decl = EvalDecl(n_envs=4, n_sentinels=1)
    table = eval_table(decl, [0, 1], ("heuristic",))
    svc = _Svc()
    assert load_fixed_slots(svc, decl, table, {}) == [] and svc.loads == []
    decl = EvalDecl(n_envs=4, fixed_labels=("ext_a",))
    table = eval_table(decl, [0, 1], ("heuristic",))
    with pytest.raises(RuntimeError, match="no policy"):
        load_fixed_slots(_Svc(), decl, table, {})
