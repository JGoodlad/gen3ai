"""``main.cycle_monitor tick`` end to end on the REAL head-to-head engine (CPU, tiny): a fake run of three
periodic checkpoints (two perturbed-fresh weight sets), a tiny (NON-registered) monitor rule and plan, one tick that
plays both checks' rows and decides the first decidable one, and a second tick that plays nothing. The rows are
``main.h2h``'s own (purpose ``monitor``, request kind ``monitor_row``, the h2h protocol, ONE family per run), the
decision is written once, ``verify`` re-derives it and ``audit`` is clean. The numbers are not the point: the
plumbing is. Lives beside ``main.h2h``'s tests for their checkpoint fixtures."""
from __future__ import annotations

import pytest

from agents.training import cycle_monitor as CM
from agents.training import eval_ledger as L
from agents.training.eval_ledger import audit as A
from main import cycle_monitor as M
from main.h2h import play as PL
from main.h2h.plateau_integration_test import as_checkpoint

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]

COMPUTE = PL.Compute(device="cpu", backend="eager", n_envs=8, threads=2, torch_threads=1, front="ffi",
                     profile="selfcheck")
RULE = CM.MonitorRule(n_boot=50, posterior_draws=5, pairs_per_edge=2, batch_pairs=2, topup_pairs=2)
PLAN = CM.MonitorPlan(delta_steps=1000, dense_lags=2, geo_lags=(), slack_steps=500)
ALL_ROWS = L.ReaderDecl(name="cycle_monitor_integration_test", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(),
                        requests="any", selection="include", flags_ok=frozenset(), inference="conditional")


@pytest.fixture(scope="module")
def ticked(built, checkpoints, tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setattr(CM, "REGISTERED", RULE)       # status / verify read the registered rule
    a, b = checkpoints
    base = tmp_path_factory.mktemp("monitor")
    run = base / "rb_monitor_test"
    as_checkpoint(b, run, 1000)
    as_checkpoint(a, run, 2000)
    as_checkpoint(b, run, 3000)
    out = base / "ledger"
    logs: list = []
    M.tick(run, [], str(out), COMPUTE, plan=PLAN, rule=RULE, emit=logs.append, inflight=1)
    rows1 = list(L.read(ALL_ROWS, root=out).rows)
    M.tick(run, [], str(out), COMPUTE, plan=PLAN, rule=RULE, emit=logs.append, inflight=1)
    rows2 = list(L.read(ALL_ROWS, root=out).rows)
    return run, out, rows1, rows2, logs


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_one_tick_plays_every_row_and_decides_once(ticked):
    run, out, rows, _rows2, _logs = ticked
    rows_main = [r for r in rows if r["request"]["kind"] == CM.REQUEST_KIND_ROW]
    edges = sorted({(r["player"]["step"], r["opponent"]["step"]) for r in rows_main})
    assert edges == [(2000, 1000), (3000, 1000), (3000, 2000)]
    for r in rows_main:
        assert r["purpose"] == "monitor" and r["request"]["family"] == M.family_id(run.name, PLAN)
        assert r["regime"]["protocol"] == M.PROTOCOL and r["seed"]["schedule_key"].startswith("monitor_row:")
    ds = L.read_decisions(root=out, kind=CM.DECISION_KIND)
    assert [d["request_id"] for d in ds] == [CM.row_request_id(run.name, 3000, PLAN)]
    assert ds[0]["verdict"] in CM.VERDICTS and ds[0]["family"] == M.family_id(run.name, PLAN)


def test_a_second_tick_plays_nothing_and_the_decision_verifies(ticked):
    _run, out, rows1, rows2, _logs = ticked
    assert [r["row_id"] for r in rows2] == [r["row_id"] for r in rows1]
    ds = L.read_decisions(root=out, kind=CM.DECISION_KIND)
    assert len(ds) == 1
    v = A.verify(out, ds[0]["decision_id"])
    assert v.ok and v.verdict_rederived == ds[0]["verdict"], (v.problems, v.note)
    assert A.audit(out).ok
