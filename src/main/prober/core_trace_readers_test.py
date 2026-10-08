"""The trace readers OUTSIDE the prober on a Rust-eval CORE trace (F-LH-5): each one EXPANDS it, reads
its stored meta, or REFUSES it by name — never reads it as empty.

A core trace (``gen3_core_trace_v1``) stores ``meta`` only in its ``*_summary.json`` and a NaN
``win_probs`` column in its ``*_states.npz`` (F-LH-4). The static gate
(``src/trace_summary_reader_gate_test.py``) holds the READ SITE to the loader; this file holds each
reader family's BEHAVIOUR on a core trace. The expansion itself (replay + cross-check) is
``core_trace_integration_test.py`` (``sim``); here it is stubbed at ``core_trace.expand`` — the one
function every expanding read goes through — so the file stays unmarked and takes milliseconds.
Every test FAILS on revert of the reader it names (a direct ``json.load`` reads zero decisions, or an
empty frame, and nothing raises).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from main.prober import core_trace
from main.prober.core_trace import CoreTraceUnsupported

STEP = 1000
N_ROWS = 5


def _write_core(run_dir: str, name: str, result: str = "LOSS", turns: int = 40) -> str:
    from agents.training.rust_eval.traces import write_core_trace

    pre = os.path.join(run_dir, "eval_traces", f"step_{STEP}", "heuristic", name)
    masks = np.ones((N_ROWS, 11), dtype=bool)
    write_core_trace(pre, trace={"records": {"p1": "", "p2": ""}, "recon": {}}, step=STEP,
                     battle_id=f"core-{STEP}-heuristic-{name}", result=result, draw_kind=None, turns=turns,
                     trainee_username="lepone", obs=np.zeros((N_ROWS, 4), dtype=np.float32),
                     logp=np.full((N_ROWS, 11), -np.log(11.0), dtype=np.float32),
                     values=np.linspace(0.2, 0.8, N_ROWS).astype(np.float32),
                     actions=np.arange(N_ROWS), masks=masks)
    return pre + "_summary.json"


def _fake_expansion(summary_path, summary, *, impl="rust", run_dir=None):
    """What the replay would rebuild: the stored meta + one ``invocations`` entry per states row."""
    acts = {m: {"valid": True, "prob": "9.1%"} for m in ("substitute", "protect")}
    acts.update({f"slot{i}": {"valid": True, "prob": "9.1%"} for i in range(9)})
    return {"meta": summary["meta"], "teams": {"p1": [], "p2": []},
            "invocations": [{"phase": "move_selection", "turn": 30 + i, "chosen": "substitute",
                             "actions": dict(acts)} for i in range(N_ROWS)]}


@pytest.fixture
def core_run(tmp_path):
    run_dir = str(tmp_path / "run")
    paths = [_write_core(run_dir, "loss_g0_001"), _write_core(run_dir, "win_g1_002", result="WIN", turns=120)]
    return run_dir, paths


@pytest.fixture
def no_expand(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("a META reader expanded a core trace (a replay per battle for one field)")

    monkeypatch.setattr(core_trace, "expand", refuse)


@pytest.fixture
def fake_expand(monkeypatch):
    monkeypatch.setattr(core_trace, "expand", _fake_expansion)


# ---------------------------------------------------------------------------
# the loader's three entry points
# ---------------------------------------------------------------------------

def test_the_loader_expands_reads_meta_and_refuses(core_run, fake_expand, tmp_path):
    _run_dir, (sp, _) = core_run
    with open(sp) as fh:
        assert "invocations" not in json.load(fh)                 # the stored summary is meta only
    assert len(core_trace.load_summary(sp)["invocations"]) == N_ROWS
    meta = core_trace.load_summary_meta(sp)
    assert meta["invocations"] == N_ROWS and meta["result"] == "LOSS"
    with pytest.raises(CoreTraceUnsupported, match="the_reader: .* CORE trace .*because"):
        core_trace.refuse_core_trace(sp, reader="the_reader", why="because")

    py = tmp_path / "py_summary.json"                              # a Python trace: all three pass it through
    py.write_text(json.dumps({"meta": {"result": "WIN"}, "invocations": [{"turn": 2}]}))
    assert core_trace.load_summary(str(py))["invocations"] == [{"turn": 2}]
    assert core_trace.load_summary_meta(str(py)) == {"result": "WIN"}
    assert core_trace.refuse_core_trace(str(py), reader="r", why="w")["invocations"] == [{"turn": 2}]


# ---------------------------------------------------------------------------
# readers that EXPAND
# ---------------------------------------------------------------------------

def test_mechanic_usage_baseline_counts_the_expanded_decisions(core_run, fake_expand):
    from agents.model.mechanic_usage_baseline import measure

    run_dir, _paths = core_run
    out = measure(run_dir)
    assert out["n_summary_files"] == 2 and out["n_decisions"] == 2 * N_ROWS   # revert: 0 decisions
    assert out["mechanics"]["substitute"]["chosen"] == 2 * N_ROWS


# ---------------------------------------------------------------------------
# readers of META only — sound on a core trace, and they never expand
# ---------------------------------------------------------------------------

def test_meta_readers_read_the_stored_outcome_without_expanding(core_run, no_expand):
    from main.critic_gate import _trace_turns
    from main.ops.quota_match import classify_on_disk

    run_dir, _paths = core_run
    split = classify_on_disk(os.path.join(run_dir, "eval_traces", f"step_{STEP}"), "heuristic")
    assert split == {"WIN": ["win_g1_002"], "LOSS": ["loss_g0_001"], "DRAW": []}
    row = _trace_turns(run_dir, stall_turns=100)[STEP]
    assert row["n_traces"] == 2 and row["n_unreadable"] == 0 and row["max_turns"] == 120
    assert row["outcomes"] == {"WIN": 1, "LOSS": 1} and row["stalls"] == 1 and row["mean_turns"] == 80


def test_harvest_meter_tail_reads_meta_and_never_reports_a_nan_phi(core_run, no_expand):
    from main.harvest_meter import _load_tail

    _run_dir, (sp, _) = core_run
    obs, meta = _load_tail(sp[: -len("_summary.json")], k=3)
    assert obs.shape == (3, 4) and meta["result"] == "loss" and meta["turns"] == 40
    assert meta["recorded_phi_T"] is None            # NaN head = not recorded, never a NaN in a mean


# ---------------------------------------------------------------------------
# readers that CANNOT be sound (they rank or gate on the recorded win-prob head) — REFUSE
# ---------------------------------------------------------------------------

def test_harvest_candidates_refuse_instead_of_an_empty_harvest(core_run, no_expand):
    from main.harvest import build_candidates

    run_dir, _paths = core_run
    models_root, run = os.path.split(run_dir)
    with pytest.raises(CoreTraceUnsupported, match="harvest.build_candidates"):
        build_candidates(models_root, [run])


def test_scaffolding_gauge_refuses_rather_than_dropping_the_cycle(core_run, no_expand):
    from main.scaffolding_gauge import collect_slices

    run_dir, _paths = core_run
    with pytest.raises(CoreTraceUnsupported, match="scaffolding_gauge.collect_slices"):
        collect_slices(run_dir)
