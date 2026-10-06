"""``main.plateau tick`` end to end on the REAL head-to-head engine (CPU, tiny): a fake run of two perturbed-fresh
checkpoints laid out as periodic checkpoints, a tiny (NON-registered) Tier-1 rule, one tick that plays the due check
batch by batch to its decision, and a second tick that plays nothing. The rows are ``main.h2h``'s own (purpose
``plateau``, request kind ``plateau_t1``, the h2h protocol), the request's spec freezes both nodes, the decision is
written once, ``verify`` re-derives it and ``audit`` is clean. The numbers are not the point: the plumbing is.
Lives beside ``main.h2h``'s tests for their checkpoint fixtures."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from agents.training import eval_ledger as L
from agents.training import plateau_t1 as T
from agents.training.eval_ledger import audit as A
from main import plateau as P
from main.h2h import play as PL

pytestmark = [pytest.mark.sim, pytest.mark.integration]

COMPUTE = PL.Compute(device="cpu", backend="eager", n_envs=8, threads=2, torch_threads=2, front="ffi",
                     profile="selfcheck")
RULE = T.Tier1Rule(batch_pairs=2, min_pairs=2, max_pairs=6)
PLAN = T.CheckPlan(delta_steps=1000, lag=1, slack_steps=500)
ALL_ROWS = L.ReaderDecl(name="plateau_integration_test", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(),
                        requests="any", selection="include", flags_ok=frozenset(), inference="conditional")


def as_checkpoint(src: str, run: Path, step: int) -> Path:
    """Copy an SB3 zip to ``run/checkpoints/checkpoint_<step>_steps.zip`` with its ``num_timesteps`` set to
    ``step`` (the node's identity the driver checks), and the run's ``model_config.json`` beside it."""
    dst = run / "checkpoints" / f"checkpoint_{step}_steps.zip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            raw = zin.read(item.filename)
            if item.filename == "data":
                data = json.loads(raw)
                data["num_timesteps"] = step
                raw = json.dumps(data).encode()
            zout.writestr(item, raw)
    shutil.copy(Path(src).parent / "model_config.json", run / "model_config.json")
    return dst


@pytest.fixture(scope="module")
def ticked(built, checkpoints, tmp_path_factory):
    a, b = checkpoints
    base = tmp_path_factory.mktemp("plateau")
    run = base / "rb_plateau_test"
    as_checkpoint(b, run, 1000)          # the W-back node
    as_checkpoint(a, run, 2000)          # the newest node
    out = base / "ledger"
    logs: list = []
    rep1 = P.tick(run, [], PLAN, str(out), COMPUTE, rule=RULE, emit=logs.append)
    rows1 = list(L.read(ALL_ROWS, root=out).rows)
    rep2 = P.tick(run, [], PLAN, str(out), COMPUTE, rule=RULE, emit=logs.append)
    rows2 = list(L.read(ALL_ROWS, root=out).rows)
    return run, out, rep1, rows1, rep2, rows2, logs


def test_one_tick_decides_the_due_check(ticked):
    run, out, rep, rows, *_ = ticked
    checks = {c["grid"]: c for c in rep["checks"]}
    assert checks[1000]["state"] == P.MISSING_NODE                       # its W-back is step 0
    c = checks[2000]
    assert c["state"] in T.VERDICTS and c["decision_id"]
    res = T.evaluate(rows, RULE)
    assert res.verdict == c["state"] and len(rows) == res.stop_batch + 1 and not res.surplus_batches
    for r in rows:
        assert r["purpose"] == "plateau" and r["request"]["kind"] == "plateau_t1"
        assert r["request"]["id"] == T.request_id(run.name, 2000, PLAN)
        assert r["regime"]["protocol"] == P.PROTOCOL and r["pairs"]["n_pairs"] + r["pairs"]["voided"] == 2
        assert r["player"]["step"] == 2000 and r["opponent"]["step"] == 1000
        assert r["seed"]["schedule_key"].startswith("plateau_t1:")


def test_a_second_tick_plays_nothing(ticked):
    _run, _out, rep1, rows1, rep2, rows2, logs = ticked
    assert [r["row_id"] for r in rows2] == [r["row_id"] for r in rows1]
    assert rep2["checks"] == rep1["checks"]
    assert any("0 due check(s)" in m for m in logs)


def test_the_decision_verifies_and_the_ledger_audits_clean(ticked):
    _run, out, rep, *_ = ticked
    d = next(c for c in rep["checks"] if c["grid"] == 2000)
    assert len(L.read_decisions(root=out, kind="plateau")) == 1
    v = A.verify(out, d["decision_id"])
    assert v.ok and v.verdict_rederived == d["state"]
    assert A.audit(out).ok
