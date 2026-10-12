"""The DEADMAN's three exits, against a FAKE API (no network): the deadline, the credit floor, and gone-already.

The live rental of 2026-10-12 exercised only the gone-already exit (the instance was destroyed by hand first), so
the two destroying exits are pinned here.
"""
import argparse
import time

import pytest

from tools.cloud.vast import api
from tools.cloud.vast import cli


class FakeVast:
    def __init__(self, ids, credit=9.0):
        self.ids, self.credit_v, self.destroyed = list(ids), credit, []

    def instances(self):
        return [{"id": i} for i in self.ids]

    def credit(self):
        return self.credit_v

    def destroy_instance(self, iid):
        self.destroyed.append(iid)
        self.ids = [i for i in self.ids if i != iid]
        return {"success": True}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeVast([42])
    for name in ("instances", "credit", "destroy_instance"):
        monkeypatch.setattr(api, name, getattr(f, name))
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    monkeypatch.setattr(cli, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "LEDGER", str(tmp_path / "ledger.jsonl"))
    return f


def run(deadline, floor=7.0):
    return cli.cmd_deadman(argparse.Namespace(instance=42, deadline=deadline, min_credit=floor))


def test_past_deadline_destroys_and_verifies(fake, tmp_path):
    assert run(time.time() - 1) == 0
    assert fake.destroyed == [42] and fake.ids == []
    assert '"by": "deadman"' in (tmp_path / "ledger.jsonl").read_text()


def test_credit_below_the_floor_destroys_before_the_deadline(fake, tmp_path):
    fake.credit_v = 6.5
    assert run(time.time() + 3600) == 0
    assert fake.destroyed == [42]
    assert "credit 6.5000 < floor 7.00" in (tmp_path / "ledger.jsonl").read_text()


def test_gone_already_exits_without_a_destroy(fake):
    fake.ids = []
    assert run(time.time() + 3600) == 0
    assert fake.destroyed == []


def test_a_transient_api_error_never_kills_the_backstop(fake, monkeypatch):
    calls = {"n": 0}
    real = fake.instances

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise api.VastError("GET /api/v1/instances/ -> HTTP 502")
        return real()

    monkeypatch.setattr(api, "instances", flaky)
    clock = iter([0.0, 1.0, 10_000.0, 10_000.0, 10_000.0])      # poll once (error), then past the deadline
    monkeypatch.setattr(cli.time, "time", lambda: next(clock, 10_000.0))
    assert run(5_000.0) == 0
    assert fake.destroyed == [42]
