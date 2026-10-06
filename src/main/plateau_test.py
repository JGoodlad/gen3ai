"""``main.plateau`` through a REAL ledger (no engine): the driver's sequential loop decides after every batch and
stops at the first decision, a decision is written ONCE (a second driver gets the first back), a crashed driver
resumes from the request's rows without replaying a batch, a changed node file is refused, ``verify`` re-derives
the verdict, and ``status`` reads every check's state and the run's Tier-1 status. The games are synthetic rows
written under claims exactly as ``main.h2h`` writes them; the engine path is ``plateau_integration_test.py``."""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.training import eval_ledger as L
from agents.training import plateau_t1 as T
from agents.training.eval_ledger import audit as A
from agents.training.eval_ledger import testkit as K
from main import plateau as P

EVEN = (0, 4, 32, 4, 0)
WIN = (0, 2, 26, 10, 2)
D = 10_000_000
#: ONE clock for every writer in a test module run (it only moves forward)
CLOCK = K.Clock()


def test_the_protocol_literal_is_main_h2h_s():
    from main.h2h import play as PL

    assert P.PROTOCOL == PL.PROTOCOL


class Fake:
    """A stand-in for the engine: ``play_upto(n)`` writes the missing batches ``< n`` from a scripted pentanomial
    list, each under its claim, and records what it was asked."""

    def __init__(self, writer, req, script, fail_at=None):
        self.w, self.req, self.script, self.fail_at = writer, req, list(script), fail_at
        self.asked, self.played = [], []

    def __call__(self, n):
        self.asked.append(n)
        have = {int(r["request"]["batch"]) for r in L.read(P.TIER1_ROWS, root=self.w.root,
                                                            request_id=self.req["request_id"]).rows}
        for b in range(n):
            if b in have:
                continue
            if self.fail_at is not None and b >= self.fail_at:
                raise KeyboardInterrupt("the driver died")
            reg = K.regime()
            self.w.clock.advance(2)                     # a row is stamped by the clock a decision's as_of reads
            c = self.w.claim(self.req["request_id"], batch=b, player=K.SHA_A, opponent=K.SHA_B,
                             regime_id=reg["regime_id"], expected_wall_s=60.0)
            row = K.make_row(self.w.next_row_id(), request=self.req, batch=b, pc=self.script[b], purpose="plateau",
                             ts=self.w.clock.iso(), cycle_seed=500 + b, schedule_key="plateau_t1:" + self.req["request_id"])
            self.w.append_row(row, c)
            self.played.append(b)


def open_req(w, rid="plateau_t1:run:60000000:w50000000", spec=None):
    return w.open_request(rid, kind=T.REQUEST_KIND, purpose=T.PURPOSE, protocol=K.PROTO, spec=spec or {"x": 1})


def writer(root, clock=None):
    return L.LedgerWriter(root, producer="h2h", clock=clock or CLOCK)


def decisions(root):
    return L.read_decisions(root=root, kind="plateau")


def test_the_driver_stops_at_the_first_decision_and_writes_it_once(tmp_path):
    root = tmp_path / "ledger"
    w = writer(root)
    req = open_req(w)
    fake = Fake(w, req, [EVEN] * 10)
    d = P.drive_check(w, root, req["request_id"], T.REGISTERED, K.SHA_A, fake, lambda _m: None)
    assert d["verdict"] == T.FLAT and fake.played == [0, 1, 2, 3, 4]       # stops at the 5th batch, plays no 6th
    assert fake.asked == [1, 2, 3, 4, 5]
    assert len(decisions(root)) == 1 and d["consumed"]["count"] == 5
    assert d["rule"] == T.REGISTERED.rule_string() and d["rule_version"] == T.RULE_VERSION
    # a SECOND driver on the decided request plays nothing and gets the first decision back
    w2 = writer(root)
    fake2 = Fake(w2, req, [EVEN] * 10)
    d2 = P.drive_check(w2, root, req["request_id"], T.REGISTERED, K.SHA_A, fake2, lambda _m: None)
    assert d2["decision_id"] == d["decision_id"] and fake2.asked == [] and len(decisions(root)) == 1
    # the ledger audits clean and verify RE-DERIVES the verdict from the rows
    w.close()
    w2.close()
    assert A.audit(root).ok
    v = A.verify(root, d["decision_id"])
    assert v.ok and v.verdict_rederived == T.FLAT, (v.problems, v.note)


def test_a_crashed_driver_resumes_without_replaying(tmp_path):
    root = tmp_path / "ledger"
    w = writer(root)
    req = open_req(w)
    with pytest.raises(KeyboardInterrupt):
        P.drive_check(w, root, req["request_id"], T.REGISTERED, K.SHA_A, Fake(w, req, [WIN] * 5, fail_at=1),
                      lambda _m: None)
    assert decisions(root) == []
    w2 = writer(root)
    fake = Fake(w2, req, [WIN] * 5)
    d = P.drive_check(w2, root, req["request_id"], T.REGISTERED, K.SHA_A, fake, lambda _m: None)
    assert d["verdict"] == T.GAIN and fake.played == [1]                   # batch 0 was banked, never replayed


def test_a_decided_but_unrecorded_test_is_recorded_without_playing(tmp_path):
    root = tmp_path / "ledger"
    w = writer(root)
    req = open_req(w)
    Fake(w, req, [WIN] * 2)(2)                                             # the rows decide; the driver died
    fake = Fake(w, req, [WIN] * 2)
    d = P.drive_check(w, root, req["request_id"], T.REGISTERED, K.SHA_A, fake, lambda _m: None)
    assert d["verdict"] == T.GAIN and fake.asked == []


def test_two_decisions_cannot_be_written_for_one_request(tmp_path):
    root = tmp_path / "ledger"
    w = writer(root)
    req = open_req(w)
    Fake(w, req, [WIN] * 2)(2)
    got = L.read(P.TIER1_ROWS, root=root, request_id=req["request_id"])
    kw = dict(kind="plateau", subject=K.SHA_A, consumed=got, verdict=T.GAIN, rule=T.REGISTERED.rule_string(),
              rule_version=T.RULE_VERSION, request_id=req["request_id"])
    w.append_decision(**kw, unique=True)
    with pytest.raises(L.DecisionExistsError) as e:
        w.append_decision(**kw, unique=True)
    assert e.value.decision["verdict"] == T.GAIN and len(decisions(root)) == 1


def test_a_changed_node_file_is_refused_by_the_request_spec(tmp_path):
    from main.h2h import play as PL

    w = writer(tmp_path / "ledger")
    chk, plan = T.Check(6 * D, 6 * D + 300, D + 180), T.DEFAULT_PLAN
    a = {"id": "r@60000300", "step": 6 * D + 300, "sha256": K.SHA_A}
    b = {"id": "r@10000180", "step": D + 180, "sha256": K.SHA_B}
    spec = PL.edge_spec(40, P.SCHEDULE_SEED, P.consumer_spec("r", chk, plan, T.REGISTERED, a, b))
    open_req(w, spec=spec)
    assert open_req(w, spec=spec)["spec"] == spec                          # idempotent
    replaced = PL.edge_spec(40, P.SCHEDULE_SEED, P.consumer_spec("r", chk, plan, T.REGISTERED, dict(a, sha256=K.SHA_C), b))
    with pytest.raises(L.RequestSpecError):
        open_req(w, spec=replaced)
    # without a consumer the spec is exactly the one every existing h2h request holds
    assert PL.edge_spec(500, 0) == {"producer": "h2h", "batch_pairs": 500, "schedule_seed": 0}


def test_the_schedule_key_is_its_own_namespace_and_ordered():
    k = P.schedule_key(K.SHA_A, K.SHA_B)
    assert k.startswith("plateau_t1:") and k != P.schedule_key(K.SHA_B, K.SHA_A)


def fake_run(tmp_path: Path, steps, name="rb_fake"):
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    for s in steps:
        (run / "checkpoints" / f"checkpoint_{s}_steps.zip").write_bytes(b"")
    (run / "checkpoints" / "checkpoint_forced_7000000_120000.zip").write_bytes(b"")
    return run


def test_status_reads_every_check(tmp_path, capsys):
    steps = [D * k + 100 for k in range(1, 9)]                             # 10M .. 80M
    run = fake_run(tmp_path, steps)
    root = tmp_path / "ledger"
    w = writer(root)
    plan = T.DEFAULT_PLAN
    # 60M: FLAT (decided); 70M: FLAT (decided); 80M: in progress
    for g, script, n in ((6 * D, [EVEN] * 5, None), (7 * D, [EVEN] * 5, None), (8 * D, [WIN], 1)):
        req = open_req(w, rid=T.request_id(run.name, g, plan))
        fake = Fake(w, req, script)
        if n is None:
            P.drive_check(w, root, req["request_id"], T.REGISTERED, K.SHA_A, fake, lambda _m: None)
        else:
            fake(n)
    w.close()
    states = P.check_states(run.name, P.node_series(run), root, plan, T.REGISTERED)
    got = {s.check.grid: s.state for s in states}
    assert got == {5 * D: P.MISSING_NODE, 6 * D: T.FLAT, 7 * D: T.FLAT, 8 * D: T.CONTINUE}
    rep = P.status_report(run.name, states, plan, T.REGISTERED, root)
    assert rep["tier1"]["status"] == T.TIER1_PLATEAU and rep["tier1"]["flat_checks"] == [6 * D, 7 * D]
    assert rep["tier1"]["pending_after"] == [8 * D] and rep["conflicts"] == []
    assert P.main(["status", str(run), "--out", str(root)]) == 0
    out = capsys.readouterr().out
    assert "TIER 1 STATUS: TIER1_PLATEAU" in out and "NOT BUILT" in out and "rb_fake@60000100" in out


def test_status_flags_two_decisions_as_a_conflict(tmp_path):
    run = fake_run(tmp_path, [D * k + 100 for k in range(1, 7)])
    root = tmp_path / "ledger"
    w = writer(root)
    req = open_req(w, rid=T.request_id(run.name, 6 * D))
    Fake(w, req, [WIN] * 2)(2)
    got = L.read(P.TIER1_ROWS, root=root, request_id=req["request_id"])
    for _ in range(2):                                                   # unique=False: what the guard prevents
        w.append_decision(kind="plateau", subject=K.SHA_A, consumed=got, verdict=T.GAIN,
                          rule=T.REGISTERED.rule_string(), rule_version=T.RULE_VERSION,
                          request_id=req["request_id"])
    w.close()
    states = P.check_states(run.name, P.node_series(run), root, T.DEFAULT_PLAN, T.REGISTERED)
    assert [s.state for s in states if s.check.grid == 6 * D] == [P.CONFLICT]
    assert P.main(["status", str(run), "--out", str(root)]) == 2


def test_node_series_lineage_and_forced_saves(tmp_path):
    parent = fake_run(tmp_path, [D + 1, 2 * D + 1], name="rb_parent")
    child = fake_run(tmp_path, [2 * D + 5, 3 * D + 5], name="rb_child")
    nodes = P.node_series(child, [parent])
    assert sorted(nodes) == [D + 1, 2 * D + 1, 2 * D + 5, 3 * D + 5]       # no forced save
    assert P.node_name(nodes, D + 1) == "rb_parent@10000001" and P.node_name(nodes, None) is None
    # on the grid, 20M's node is the FIRST at or above it: the parent's 20,000,001
    assert T.node_at(2 * D, sorted(nodes), T.DEFAULT_PLAN.slack_steps) == 2 * D + 1
