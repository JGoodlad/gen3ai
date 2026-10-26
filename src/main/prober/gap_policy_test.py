"""THE OBS-SEMANTICS GAP POLICY (`arch_status.decide`, the table `GAP_POLICY`), and every caller's declared use.

A checkpoint recorded below `OBS_SEMANTICS_VERSION` has weights that fit and inputs whose meaning moved. What a reader
may do with it depends on what it feeds the network: a TRACE REPLAY (stored observations, read against what the model
recorded) is refused; a FRESH probe (positions encoded now) or a STORED-obs score (no faithfulness claim) runs with a
badge. This file pins the table cell by cell, the decision's record, and — by AST, so a new caller fails until it is
classified — the use every `ProbeModel.load` / `ProbeSession._model_for` / `probe_model` call site declares. The
behaviour at the model seam and the session is `model_test.py`'s.

Pure: no torch, no checkpoint.
"""
from __future__ import annotations

import ast
import json
import os

import pytest

from agents.model.model_version import (
    ARCH_SIGNATURE,
    MIGRATION_FLOOR,
    MODEL_CONFIG_VERSION,
    OBS_SEMANTICS_HISTORY,
    OBS_SEMANTICS_VERSION,
)
from main.prober import arch_status as A
from utils.paths import src_path

# The table as the owner's ruling reads it (2026-10-26). Changing a cell is a policy change: change it here too.
_EXPECTED = {
    (A.GAP_NONE, A.USE_TRACE_REPLAY): A.ALLOW,
    (A.GAP_NONE, A.USE_FRESH): A.ALLOW,
    (A.GAP_NONE, A.USE_STORED_SCORE): A.ALLOW,
    (A.GAP_UNRECORDED, A.USE_TRACE_REPLAY): A.WARN,
    (A.GAP_UNRECORDED, A.USE_FRESH): A.WARN,
    (A.GAP_UNRECORDED, A.USE_STORED_SCORE): A.WARN,
    (A.GAP_SEMANTICS, A.USE_TRACE_REPLAY): A.REFUSE,
    (A.GAP_SEMANTICS, A.USE_FRESH): A.WARN,
    (A.GAP_SEMANTICS, A.USE_STORED_SCORE): A.WARN,
    (A.GAP_WEIGHTS, A.USE_TRACE_REPLAY): A.REFUSE,
    (A.GAP_WEIGHTS, A.USE_FRESH): A.REFUSE,
    (A.GAP_WEIGHTS, A.USE_STORED_SCORE): A.REFUSE,
}

_HEAD_SIG = ARCH_SIGNATURE


def _verdict(gap: str) -> A.ArchVerdict:
    """One verdict per row of the table, built from a RECORD the way every caller's verdict is."""
    if gap == A.GAP_NONE:
        return A.verdict_from_record(MODEL_CONFIG_VERSION, _HEAD_SIG)
    if gap == A.GAP_UNRECORDED:
        return A.verdict_from_record(None, None)
    if gap == A.GAP_SEMANTICS:
        return A.verdict_from_record(OBS_SEMANTICS_VERSION - 1, _HEAD_SIG)
    return A.verdict_from_record(143, "gen3_event_record_v2")


def test_the_table_has_every_cell_and_each_is_the_ruled_one():
    assert set(A.GAP_POLICY) == set(A.GAPS)
    for gap in A.GAPS:
        assert set(A.GAP_POLICY[gap]) == set(A.USES), f"row {gap} is missing a use"
    assert {(g, u): A.GAP_POLICY[g][u] for g in A.GAPS for u in A.USES} == _EXPECTED


@pytest.mark.parametrize("gap,use", sorted(_EXPECTED))
def test_decide_reads_the_table_for_a_real_verdict(gap, use):
    v = _verdict(gap)
    assert A.gap_of(v) == gap
    d = A.decide(v, use)
    assert d.action == _EXPECTED[(gap, use)] and d.gap == gap and d.use == use
    assert (d.badge is not None) == (d.action == A.WARN), "a badge exactly when the policy warns"
    assert d.refused == (d.action == A.REFUSE)
    rec = d.as_dict()
    json.dumps(rec)
    assert rec["code_semantics"] == OBS_SEMANTICS_VERSION


def test_every_incompatible_kind_but_semantics_is_a_weights_gap():
    for kind in A.KINDS:
        v = A.ArchVerdict(A.INCOMPATIBLE, kind, "x")
        assert A.gap_of(v) == (A.GAP_SEMANTICS if kind == A.KIND_OBS_SEMANTICS else A.GAP_WEIGHTS)


def test_an_undeclared_use_raises():
    with pytest.raises(ValueError, match="declare one of"):
        A.decide(_verdict(A.GAP_NONE), "analyze")


def test_the_trained_semantics_is_the_newest_marker_at_or_below_the_recorded_version():
    first = OBS_SEMANTICS_HISTORY[0][0]
    assert A.obs_semantics_of(None) is None
    assert A.obs_semantics_of(first - 1) is None                    # below every declared marker
    for i, (v, _reason) in enumerate(OBS_SEMANTICS_HISTORY):
        assert A.obs_semantics_of(v) == v
        nxt = OBS_SEMANTICS_HISTORY[i + 1][0] if i + 1 < len(OBS_SEMANTICS_HISTORY) else MODEL_CONFIG_VERSION + 1
        assert A.obs_semantics_of(nxt - 1) == v
    assert [c["version"] for c in A.semantics_changes_since(first)] == [v for v, _ in OBS_SEMANTICS_HISTORY[1:]]


def test_the_badge_says_trained_n_probed_at_m():
    d = A.decide(A.verdict_from_record(OBS_SEMANTICS_VERSION - 1, _HEAD_SIG), A.USE_FRESH)
    n = A.obs_semantics_of(OBS_SEMANTICS_VERSION - 1)
    assert f"trained on obs_semantics v{n}, probed at v{OBS_SEMANTICS_VERSION}" in d.badge
    pre = A.decide(A.verdict_from_record(MIGRATION_FLOOR, _HEAD_SIG), A.USE_FRESH)
    if A.obs_semantics_of(MIGRATION_FLOOR) is None:
        assert f"before v{OBS_SEMANTICS_HISTORY[0][0]}" in pre.badge
        assert len(pre.as_dict()["changes_since"]) == len(OBS_SEMANTICS_HISTORY)


# -- every caller declares its use (AST over src/, tests excluded) ------------------------------------------------

#: (module path under src/, enclosing function) -> the use its call declares. A NEW call site fails
#: `test_every_model_loading_call_site_declares_the_use_this_table_names` until it is classified here.
_CALLERS = {
    ("main/probe_replay.py", "main"): A.USE_TRACE_REPLAY,
    ("main/harvest.py", "score_candidates"): A.USE_STORED_SCORE,
    ("main/harvest_meter.py", "main"): A.USE_STORED_SCORE,
    ("agents/training/winprob_finetune.py", "main"): A.USE_STORED_SCORE,
    ("main/prober/session/core.py", "probe_model"): "use",                 # forwards its caller's declared use
    ("main/prober/session/core.py", "_model_for"): "use",
    ("main/prober/session/analysis.py", "analyze"): A.USE_TRACE_REPLAY,
    ("main/prober/session/analysis.py", "find"): A.USE_TRACE_REPLAY,
    ("main/prober/session/game.py", "_capture"): A.USE_TRACE_REPLAY,
    ("main/prober/session/probes.py", "probe"): A.USE_TRACE_REPLAY,
    ("main/prober/session/probes.py", "history_saliency"): A.USE_TRACE_REPLAY,
    ("main/prober/session/counterfactual.py", "lookahead"): A.USE_FRESH,
    ("main/prober/session/counterfactual.py", "better_line"): A.USE_FRESH,
}

_CONSTS = {"USE_TRACE_REPLAY": A.USE_TRACE_REPLAY, "USE_FRESH": A.USE_FRESH, "USE_STORED_SCORE": A.USE_STORED_SCORE,
           "_TRACE_REPLAY": A.USE_TRACE_REPLAY, "_FRESH": A.USE_FRESH}


def _call_name(node: ast.Call) -> "str | None":
    f = node.func
    if isinstance(f, ast.Attribute):
        if f.attr == "load" and isinstance(f.value, ast.Name) and f.value.id == "ProbeModel":
            return "ProbeModel.load"
        if f.attr in ("_model_for", "probe_model"):
            return f.attr
    return None


def _use_value(node: ast.Call) -> "str | None":
    for kw in node.keywords:
        if kw.arg == "use":
            v = kw.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return v.value
            if isinstance(v, ast.Name):
                return _CONSTS.get(v.id, v.id)
            if isinstance(v, ast.Attribute):
                return _CONSTS.get(v.attr, v.attr)
            return "<expr>"
    return None


def _call_sites() -> "dict[tuple[str, str], set[str | None]]":
    root = str(src_path())
    out: dict = {}
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "rust_sim", "node_modules")]
        for fn in files:
            if not fn.endswith(".py") or fn.endswith("_test.py") or fn.startswith("test_"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(path).read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            rel = os.path.relpath(path, root)
            for func in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
                for node in ast.walk(func):
                    if isinstance(node, ast.Call) and _call_name(node):
                        out.setdefault((rel, func.name), set()).add(_use_value(node))
    return out


def test_every_model_loading_call_site_declares_the_use_this_table_names():
    sites = _call_sites()
    # An enclosing function nested in another is walked twice (both scopes see the call); key on the innermost
    # names this table uses, and require every table row to be found.
    missing = [k for k in _CALLERS if k not in sites]
    assert not missing, f"a classified caller no longer loads a model (re-classify the table): {missing}"
    unclassified = sorted(k for k in sites if k not in _CALLERS)
    assert not unclassified, (f"new model-loading call sites — classify each as trace_replay / fresh / stored_score "
                              f"in _CALLERS (designs/prober/arch_drift.md): {unclassified}")
    wrong = {k: sorted(map(str, v)) for k, v in sites.items() if v != {_CALLERS[k]}}
    assert not wrong, f"a call site's declared use differs from its classification: {wrong}"


def test_the_counterfactual_play_out_and_the_harvest_rollouts_decide_fresh_from_the_records():
    """The two FRESH paths that never build a ProbeModel (bare MaskablePPO rollouts) still ask the policy."""
    src = open(os.path.join(str(src_path()), "main", "prober", "session", "counterfactual.py")).read()
    assert "A.decide(v, _FRESH)" in src and "_play_semantics(choice.path)" in src
    hsrc = open(os.path.join(str(src_path()), "main", "harvest.py")).read()
    assert "A.decide(v, A.USE_FRESH)" in hsrc and '"rollout_obs_semantics": _rollout_semantics(subject)' in hsrc
