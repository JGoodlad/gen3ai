"""END TO END, no game played: a spec with one S group goes plan -> run (the S lane, its heavy-slot unit replaced by
a stub that appends the cells' ledger rows) -> assemble -> decide (``x35_r1``) -> record, and the record carries the
ledger's regime id and digest; R moving under the read stops the lane (exit 3), never a silent re-score."""
from __future__ import annotations

import hashlib
import json
import re
import time

import pytest

from agents.training import eval_ledger as L
from agents.training.eval_ledger import testkit as K
from main.reads import api, layout as LY
from main.reads import runner as RN
from main.reads.models import sha256_file
from main.x35 import refpop as RP

CODE = {"commit": "a" * 40, "dirty": False, "root": "/tmp/pin"}
SEEDS = (3501, 3502, 3503)


def _zip(d, data):
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_config.json").write_text("{}")
    z = d / "final_model.zip"
    z.write_bytes(data)
    return z


def _refpop(tmp_path):
    ms = [{"label": f"m{i}", "block": "A", "path": str(_zip(tmp_path / "r" / f"m{i}", f"member{i}".encode()))}
          for i in (1, 2)]
    for m in ms:
        m["sha256"] = sha256_file(m["path"])
    doc = {"schema": RP.SCHEMA, "pstar": dict(ms[0], role="pstar"), "archive": [], "a_block": [dict(ms[1])],
           "x_block": [], "members": ms}
    doc["content_sha256"] = hashlib.sha256(RP._canon(doc)).hexdigest()
    p = tmp_path / "refpop.json"
    p.write_text(json.dumps(doc))
    return p, doc


def _void(tmp_path, k9b="PASS", verdict_line=True):
    """SPRT's registered VOID gate inputs: each ON seed's run dir (``final_model.zip`` + ``sprt_promotion.jsonl``) and
    its AM-6 K9(b) record. ``k9b`` = the verdict seed 3502's record reads; no ``verdict_line`` = V-S1 fires there."""
    kdir = tmp_path / "k9b"
    kdir.mkdir(exist_ok=True)
    runs = {}
    for s in SEEDS:
        rd = tmp_path / "runs" / f"rb_x35_sprt_s{s}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "final_model.zip").write_bytes(b"z")
        lines = [{"step": 1, "event": "start"}]
        if verdict_line or s != 3502:
            lines.append({"step": 1, "event": "verdict", "promoted": True, "reason": "accept"})
        (rd / "sprt_promotion.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        (kdir / f"{rd.name}.json").write_text(json.dumps({"verdict": k9b if s == 3502 else "PASS"}))
        runs[f"sprt_s{s}"] = str(rd)
    return {"k9b_dir": str(kdir), "runs": runs}


def _spec(tmp_path, refpop, content, void=None):
    models = {}
    for arm in ("sprt", "c0"):
        for s in SEEDS:
            z = _zip(tmp_path / "m" / f"{arm}_{s}", f"{arm}{s}".encode())
            models[f"{arm}_s{s}"] = {"arm": arm, "seed": s, "ref": {"kind": "pinned", "path": str(z),
                                                                    "sha256": sha256_file(z)}}
    doc = {"schema": "gen3_read_spec_v1", "read": "demo_s_read_2026-10-25", "models": models,
           "s": {"pool": {"refpop": str(refpop), "refpop_content_sha256": content, "panel": None, "pairs": 500,
                          "batch_pairs": 100, "family": "demo_s_2026-10-25", "models": sorted(models)}},
           "decision": {"rule": "x35_r1", "params": {"arm": "SPRT", "look": 1, "on": "sprt", "off": "c0",
                                                     "s_group": "pool", "void": void or _void(tmp_path)}}}
    p = tmp_path / "read_spec.json"
    p.write_text(json.dumps(doc))
    return p


class FakeSlot:
    """Stands in for ``mem_cap.sh --slot ... python -m main.reads s-unit``: appends 100-pair batches for every cell
    the engines name, as ``main.h2h play-many`` would (one row per batch)."""

    def __init__(self, state, refpop_doc):
        self.state, self.calls = state, []
        self.members = {m["label"]: m["sha256"] for m in refpop_doc["members"]}

    def __call__(self, argv):
        self.calls.append(argv)
        i = argv.index("s-unit")
        a = argv[i + 1:]
        g = a[a.index("--group") + 1]
        engines = [a[j + 1] for j, x in enumerate(a) if x == "--engine"]
        models = json.loads(LY.models_file(self.state).read_text())
        spec = RN.load_state(self.state)
        x = spec.s[g]
        clock = K.Clock()
        clock.advance(len(self.calls) * 3600)
        w = L.LedgerWriter(LY.s_ledger(self.state, g), "h2h", clock=clock, host="testhost", pid=11,
                           alive=lambda _p: True)
        reg = K.regime()
        for e in engines:
            label, labs = e.split(":", 1)
            req = w.open_request(x.request(label), kind="ab_cell", purpose="ab", family=x.family, spec={})
            sh = models[label]["sha256"]
            seed = SEEDS.index(spec.models[label].seed)
            for m in labs.split(","):
                for b in range(x.pairs // x.batch_pairs):
                    clock.advance(60)
                    c = w.claim(req["request_id"], batch=b, player=sh, opponent=self.members[m],
                                regime_id=reg["regime_id"], expected_wall_s=60)
                    pc = (15 + seed, 20, 25 - seed, 20, 20) if spec.models[label].arm == "sprt" else (20,) * 5
                    w.append_row(K.make_row(w.next_row_id(), request=req, batch=b, p=sh, o=self.members[m], reg=reg,
                                            pc=pc, ts=clock.iso(), purpose="ab",
                                            cycle_seed=hash((label, m, b)) % 10 ** 9), c)
        return 0


def test_plan_run_assemble_decide_record_end_to_end(tmp_path, no_git):
    refpop, rdoc = _refpop(tmp_path)
    spec = _spec(tmp_path, refpop, rdoc["content_sha256"])
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=lambda _p: 300_000_000)
    slot = FakeSlot(state, rdoc)
    assert RN.run(state, "s", time.time() + 7200, run_s=slot, sleep=lambda s: None) == 0
    assert len(slot.calls) == 6, "a unit splits ONE node's 2 member cells over its 2 engines: 6 nodes, 6 units"
    assert all(c[0].endswith("mem_cap.sh") and c[1] == "--slot" for c in slot.calls)
    assert RN.run(state, "s", time.time() + 7200, run_s=slot, sleep=lambda s: None) == 0
    assert len(slot.calls) == 6, "a recorded batch is never replayed"
    rec = api.record(state)
    assert rec["regimes"] == {"anchor": {}, "s": {"pool": K.regime()["regime_id"]}}
    assert rec["ledgers"]["pool"]["digest"] and rec["cells_inconclusive"] == []
    assert rec["arm"] == "SPRT" and rec["look"] == 1 and rec["verdict"] == rec["decision"]["verdict"]
    assert set(rec["r1"]["seeds"]) == {"3501", "3502", "3503"}
    assert api.load_record(LY.record_file(state))["read"] == "demo_s_read_2026-10-25"
    on = json.loads(LY.inputs_file(state).read_text())["r1"]["on"]
    assert all(isinstance(v, float) for v in on.values())
    assert set(rec["void"]) == {"sprt_s3501", "sprt_s3502", "sprt_s3503"} and all(g["pass"] for g in rec["void"].values())


@pytest.mark.parametrize("void_kw,needle", [({"k9b": "FAIL"}, "K9(b) FAIL"),
                                            ({"verdict_line": False}, "SPRT V-S1/V-S2 VOID")])
def test_the_registered_void_gate_runs_before_the_record(tmp_path, no_git, void_kw, needle):
    """be60f24f FINDING 3: SPRT's assembly STOPPED on a seed whose K9(b) or V-S1 / V-S2 was not PASS; the module's
    record did not. Now ``decide`` (and so ``record``) refuses, and names the seed."""
    refpop, rdoc = _refpop(tmp_path)
    spec = _spec(tmp_path, refpop, rdoc["content_sha256"], void=_void(tmp_path, **void_kw))
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=lambda _p: 300_000_000)
    assert RN.run(state, "s", time.time() + 7200, run_s=FakeSlot(state, rdoc), sleep=lambda s: None) == 0
    with pytest.raises(api.RuleInputError, match=rf"VOID gate fails.*sprt_s3502.*{re.escape(needle)}"):
        api.record(state)
    assert not LY.record_file(state).exists() and not LY.decision_file(state).exists()
    gate = json.loads(LY.void_file(state).read_text())
    assert [lab for lab, g in gate.items() if not g["pass"]] == ["sprt_s3502"]


def test_an_sprt_spec_without_its_void_gate_is_refused(tmp_path):
    from main.reads.spec import SpecError, load

    refpop, rdoc = _refpop(tmp_path)
    p = _spec(tmp_path, refpop, rdoc["content_sha256"])
    doc = json.loads(p.read_text())
    del doc["decision"]["params"]["void"]
    p.write_text(json.dumps(doc))
    with pytest.raises(SpecError, match="void: required"):
        load(p)


def test_r_moving_under_the_s_lane_stops_it(tmp_path, no_git):
    refpop, rdoc = _refpop(tmp_path)
    spec = _spec(tmp_path, refpop, rdoc["content_sha256"])
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=lambda _p: 300_000_000)
    with open(rdoc["members"][1]["path"], "ab") as f:
        f.write(b"moved")
    slot = FakeSlot(state, rdoc)
    assert RN.run(state, "s", time.time() + 7200, run_s=slot, sleep=lambda s: None) == RN.R_MOVED
    assert slot.calls == [], "nothing is played on a moved R"
    log = [json.loads(ln) for ln in (state / "s_supervisor.jsonl").read_text().splitlines()]
    assert log[-1]["event"] == "r_moved" and "does not verify" in log[-1]["why"]


def test_a_changed_r_file_also_stops_the_lane(tmp_path, no_git):
    refpop, rdoc = _refpop(tmp_path)
    spec = _spec(tmp_path, refpop, rdoc["content_sha256"])
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=lambda _p: 300_000_000)
    doc = json.loads(refpop.read_text())
    doc["note"] = "re-frozen"
    doc["content_sha256"] = hashlib.sha256(RP._canon(doc)).hexdigest()
    refpop.write_text(json.dumps(doc))
    assert RN.run(state, "s", time.time() + 7200, run_s=FakeSlot(state, rdoc), sleep=lambda s: None) == RN.R_MOVED


def test_a_unit_that_adds_no_pair_is_a_named_refusal_not_a_loop(tmp_path, no_git):
    refpop, rdoc = _refpop(tmp_path)
    spec = _spec(tmp_path, refpop, rdoc["content_sha256"])
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=lambda _p: 300_000_000)
    calls = []
    with pytest.raises(RN.RunError, match="no new pair"):
        RN.run(state, "s", time.time() + 7200, run_s=lambda argv: calls.append(argv) or 0, sleep=lambda s: None)
    assert len(calls) == 1


def test_an_engine_that_fails_on_its_own_is_reported_even_when_another_hits_the_box(tmp_path, no_git, monkeypatch):
    import sys

    refpop, rdoc = _refpop(tmp_path)
    spec = _spec(tmp_path, refpop, rdoc["content_sha256"])
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=lambda _p: 300_000_000)

    def argv(spec, state, g, label, members, models, mem, *, threads, py=sys.executable):
        code = "raise SystemExit(7)" if label == "c0_s3501" else "import time; time.sleep(30)"
        return [sys.executable, "-c", code]

    monkeypatch.setattr(RN, "play_many_argv", argv)
    assert RN.s_unit(state, "pool", ["c0_s3501:m1", "c0_s3502:m1"], 3) == 7
    assert RN.s_unit(state, "pool", ["c0_s3502:m1"], 1) == RN.UNIT_TIME_UP
