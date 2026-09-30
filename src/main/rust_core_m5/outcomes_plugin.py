"""A pytest plugin (``-p main.rust_core_m5.outcomes_plugin``): every test's verdict, by node id,
written as JSON to ``$GEN3AI_M5J_OUTCOMES`` at session end — the gate runner's input.

The per-phase fold is ``utils.slow_tier_status``'s (``classify`` / ``merge_outcome`` / ``settle``),
so a verdict here means what it means in the slow tier: a TIMEOUT is ``inconclusive``, never
``fail``; a test with no CALL report cannot be ``pass``. Under xdist only the controller writes
(reports are forwarded to it). A collection error is a ``fail`` row keyed by the file.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

ENV = "GEN3AI_M5J_OUTCOMES"

_state: Dict[str, Any] = {"rows": {}, "calls": set(), "config": None}


def pytest_configure(config):
    _state["config"] = config


def _controller() -> bool:
    cfg = _state["config"]
    return cfg is not None and not hasattr(cfg, "workerinput")


def pytest_runtest_logreport(report):
    from utils.slow_tier_status import classify, merge_outcome

    if not _controller():
        return
    text = str(getattr(report, "longrepr", "") or "")
    status = classify(report.failed, report.skipped, text)
    rows = _state["rows"]
    rows[report.nodeid] = merge_outcome(rows.get(report.nodeid), status)
    if report.when == "call":
        _state["calls"].add(report.nodeid)


def pytest_collectreport(report):
    if _controller() and report.failed:
        _state["rows"][report.nodeid or "<collection>"] = "fail"


def pytest_sessionfinish(session, exitstatus):
    from utils.slow_tier_status import settle

    path = os.environ.get(ENV)
    if not path or not _controller():
        return
    rows = {k: settle(v, k in _state["calls"]) if v == "pass" else v for k, v in _state["rows"].items()}
    with open(path, "w") as f:
        json.dump({"exitstatus": int(exitstatus), "outcomes": rows}, f, indent=1, sort_keys=True)
